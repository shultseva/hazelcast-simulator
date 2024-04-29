package com.hazelcast.simulator.tests.vector;

import com.hazelcast.config.vector.Metric;
import com.hazelcast.config.vector.VectorCollectionConfig;
import com.hazelcast.config.vector.VectorIndexConfig;
import com.hazelcast.core.Pipelining;
import com.hazelcast.simulator.hz.HazelcastTest;
import com.hazelcast.simulator.test.BaseThreadState;
import com.hazelcast.simulator.test.annotations.AfterRun;
import com.hazelcast.simulator.test.annotations.Setup;
import com.hazelcast.simulator.test.annotations.TimeStep;
import com.hazelcast.vector.SearchOptions;
import com.hazelcast.vector.SearchOptionsBuilder;
import com.hazelcast.vector.VectorCollection;
import com.hazelcast.vector.VectorDocument;
import com.hazelcast.vector.VectorValues;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.atomic.AtomicInteger;

import static java.lang.String.format;
import static java.util.concurrent.TimeUnit.NANOSECONDS;

public class VectorCollectionSearchDatasetTest extends HazelcastTest {

    public String datasetUrl;

    public String workingDirectory;

    // common parameters
    public int numberOfSearchIterations = Integer.MAX_VALUE;

    public int loadFirst = Integer.MAX_VALUE;

    // graph parameters
    public String metric = "COSINE";

    public int maxDegree = 40;

    public int efConstruction = 50;

    // search parameters

    public int limit = 1;

    // maximum concurrent putAllAsync operations in flight
    public int maxPutAllInFlight = 10;

    // inner test parameters

    private static final String collectionName = "performance-collection";

    private static final int PUT_BATCH_SIZE = 10_000;
    private VectorCollection<Integer, Integer> collection;

    private final AtomicInteger searchCounter = new AtomicInteger(0);

    private final AtomicInteger putCounter = new AtomicInteger(0);
    private float[][] testDataset;

    @Setup
    public void setup() {
        DatasetReader reader = DatasetReader.create(datasetUrl, workingDirectory);
        var size = Math.min(reader.getSize(), loadFirst);
        int dimension = reader.getDimension();
        assert dimension == reader.getTestDatasetDimension() : "dataset dimension does not correspond to query vector dimension";
        testDataset = reader.getTestDataset();
        numberOfSearchIterations = Math.min(numberOfSearchIterations, testDataset.length);

        collection = VectorCollection.getCollection(
                targetInstance,
                new VectorCollectionConfig(collectionName)
                        .addVectorIndexConfig(
                                new VectorIndexConfig()
                                        .setMetric(Metric.valueOf(metric))
                                        .setDimension(dimension)
                                        .setMaxDegree(maxDegree)
                                        .setEfConstruction(efConstruction)
                        )
        );

        var start = System.nanoTime();

        Map<Integer, VectorDocument<Integer>> buffer = new HashMap<>(PUT_BATCH_SIZE);
        int index;
        logger.info("Start loading data...");
        Pipelining<Void> pipelining = new Pipelining<>(maxPutAllInFlight);
        while ((index = putCounter.getAndIncrement()) < size) {
            buffer.put(index, VectorDocument.of(index, VectorValues.of(reader.getTrainVector(index))));
            if (buffer.size() % PUT_BATCH_SIZE == 0) {
                // send a putAll batch
                logger.info("Sending a putAll batch");
                addToPipelineWithLogging(pipelining, collection.putAllAsync(buffer));
                // Clearing the buffer currently only works with a Hazelcast Client as test driver.
                // When test executes on an embedded member, the map that is passed as argument
                // to putAllAsync is used by the implementation. Until we fix this by making a defensive copy
                // of the map, we should be creating a new buffer HashMap here instead of clear()
                // buffer.clear();
                buffer = new HashMap<>(PUT_BATCH_SIZE);
            }
        }
        if (!buffer.isEmpty()) {
            addToPipelineWithLogging(pipelining, collection.putAllAsync(buffer));
            logger.info(format("Uploaded vectors. Last block size: %s.", buffer.size()));
            buffer.clear();
        }
        // block until all putAll batches are done
        try {
            pipelining.results();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        logger.info("Finished loading data after ", NANOSECONDS.toSeconds(System.nanoTime() - start));

        var startCleanup = System.nanoTime();
        collection.optimizeAsync().toCompletableFuture().join();

        logger.info("Collection size: " + size);
        logger.info("Collection dimension: " + reader.getDimension());
        logger.info("Cleanup time(ms): " + NANOSECONDS.toMillis (System.nanoTime() - startCleanup));
        logger.info("Index build time(m): " + NANOSECONDS.toMinutes(System.nanoTime() - start));
    }

    @TimeStep(prob = 1)
    public void search(ThreadState state) {
        var iteration = searchCounter.incrementAndGet();
        if (iteration >= numberOfSearchIterations) {
            testContext.stop();
            return;
        }
        var vector = testDataset[iteration];
        SearchOptions options = new SearchOptionsBuilder().vector(vector).includePayload().limit(limit).build();
        var result = collection.searchAsync(options).toCompletableFuture().join();

        var score = result.results().next().getScore();
        ScoreMetrics.set((int) (score * 100));
        logger.info("Found score: " + result.results().next().getScore());
    }

    @AfterRun
    public void afterRun() {
        logger.info("Number of search iteration: " + searchCounter.get());
        logger.info("Min score: " + ScoreMetrics.getMin());
        logger.info("Max score: " + ScoreMetrics.getMax());
        logger.info("Mean score: " + ScoreMetrics.getMean());
        logger.info("Percent lower then 0.98: " + ScoreMetrics.getPercentLowerThen(98));
    }

    public static class ThreadState extends BaseThreadState {
    }

    void addToPipelineWithLogging(Pipelining<Void> pipelining, CompletionStage<Void> asyncInvocation) {
        var now = System.nanoTime();
        try {
            pipelining.add(asyncInvocation);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        var nanosBlocked = System.nanoTime() - now;
        // log if we were blocked for more than 10ms
        if (now > 10_000_000) {
            logger.info(format("Thread was blocked for %d msec due to reaching max pipeline depth",
                    NANOSECONDS.toMillis(nanosBlocked)));
        }
    }
}
