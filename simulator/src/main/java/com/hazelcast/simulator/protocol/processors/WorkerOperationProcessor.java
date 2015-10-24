/*
 * Copyright (c) 2008-2015, Hazelcast, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.hazelcast.simulator.protocol.processors;

import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.simulator.protocol.core.ResponseType;
import com.hazelcast.simulator.protocol.core.SimulatorAddress;
import com.hazelcast.simulator.protocol.exception.ExceptionLogger;
import com.hazelcast.simulator.protocol.operation.CreateTestOperation;
import com.hazelcast.simulator.protocol.operation.OperationType;
import com.hazelcast.simulator.protocol.operation.PhaseCompletedOperation;
import com.hazelcast.simulator.protocol.operation.SimulatorOperation;
import com.hazelcast.simulator.protocol.operation.StartTestOperation;
import com.hazelcast.simulator.protocol.operation.StartTestPhaseOperation;
import com.hazelcast.simulator.protocol.operation.StopTestOperation;
import com.hazelcast.simulator.test.TestCase;
import com.hazelcast.simulator.test.TestPhase;
import com.hazelcast.simulator.worker.TestContainer;
import com.hazelcast.simulator.worker.TestContextImpl;
import com.hazelcast.simulator.worker.Worker;
import com.hazelcast.simulator.worker.WorkerType;
import org.apache.log4j.Logger;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import static com.hazelcast.simulator.protocol.core.ResponseType.SUCCESS;
import static com.hazelcast.simulator.protocol.core.ResponseType.UNSUPPORTED_OPERATION_ON_THIS_PROCESSOR;
import static com.hazelcast.simulator.protocol.core.SimulatorAddress.COORDINATOR;
import static com.hazelcast.simulator.utils.FileUtils.isValidFileName;
import static com.hazelcast.simulator.utils.PropertyBindingSupport.bindProperties;
import static com.hazelcast.simulator.utils.TestUtils.getUserContextKeyFromTestId;
import static java.lang.String.format;

/**
 * An {@link OperationProcessor} implementation to process {@link SimulatorOperation} instances on a Simulator Worker.
 */
public class WorkerOperationProcessor extends OperationProcessor {

    private static final String DASHES = "---------------------------";
    private static final Logger LOGGER = Logger.getLogger(WorkerOperationProcessor.class);

    private final AtomicInteger testsPending = new AtomicInteger(0);
    private final AtomicInteger testsCompleted = new AtomicInteger(0);

    private final ConcurrentMap<String, TestContainer> tests = new ConcurrentHashMap<String, TestContainer>();
    private final ConcurrentMap<String, TestPhase> testPhases = new ConcurrentHashMap<String, TestPhase>();

    private final ExceptionLogger exceptionLogger;

    private final WorkerType type;
    private final HazelcastInstance hazelcastInstance;
    private final Worker worker;

    public WorkerOperationProcessor(ExceptionLogger exceptionLogger, WorkerType type, HazelcastInstance hazelcastInstance,
                                    Worker worker) {
        super(exceptionLogger);
        this.exceptionLogger = exceptionLogger;

        this.type = type;
        this.hazelcastInstance = hazelcastInstance;
        this.worker = worker;
    }

    public Collection<TestContainer> getTests() {
        return tests.values();
    }

    TestPhase getTestPhase(String testId) {
        return testPhases.get(testId);
    }

    @Override
    protected ResponseType processOperation(OperationType operationType, SimulatorOperation operation,
                                            SimulatorAddress sourceAddress) throws Exception {
        switch (operationType) {
            case TERMINATE_WORKERS:
                processTerminateWorkers();
                break;
            case CREATE_TEST:
                processCreateTest((CreateTestOperation) operation);
                break;
            case START_TEST_PHASE:
                processStartTestPhase((StartTestPhaseOperation) operation);
                break;
            case START_TEST:
                processStartTest((StartTestOperation) operation);
                break;
            case STOP_TEST:
                processStopTest((StopTestOperation) operation);
                break;
            default:
                return UNSUPPORTED_OPERATION_ON_THIS_PROCESSOR;
        }
        return SUCCESS;
    }

    private void processTerminateWorkers() {
        worker.shutdown();
    }

    private void processCreateTest(CreateTestOperation operation) throws Exception {
        TestCase testCase = operation.getTestCase();
        String testId = testCase.getId();
        if (tests.containsKey(testId)) {
            throw new IllegalStateException(format("Can't init TestCase: %s, another test with testId [%s] already exists",
                    operation, testId));
        }
        if (!testId.isEmpty() && !isValidFileName(testId)) {
            throw new IllegalArgumentException(format("Can't init TestCase: %s, testId [%s] is an invalid filename",
                    operation, testId));
        }

        LOGGER.info(format("%s Initializing test %s %s%n%s", DASHES, testId, DASHES, testCase));

        Object testInstance = CreateTestOperation.class.getClassLoader().loadClass(testCase.getClassname()).newInstance();
        bindProperties(testInstance, testCase, TestContainer.OPTIONAL_TEST_PROPERTIES);
        TestContextImpl testContext = new TestContextImpl(testId, hazelcastInstance);

        tests.put(testId, new TestContainer(testInstance, testContext, testCase));
        testsPending.incrementAndGet();

        if (type == WorkerType.MEMBER) {
            hazelcastInstance.getUserContext().put(getUserContextKeyFromTestId(testId), testInstance);
        }
    }

    private void processStartTestPhase(StartTestPhaseOperation operation) throws Exception {
        final String testId = operation.getTestId();
        final TestPhase testPhase = operation.getTestPhase();

        try {
            final TestContainer test = tests.get(testId);
            if (test == null) {
                // we log a warning: it could be that it's a newly created machine from mama-monkey
                LOGGER.warn(format("Failed to process operation %s, found no test with testId %s", operation, testId));
                return;
            }

            OperationThread operationThread = new OperationThread(testId, testPhase) {
                @Override
                public void doRun() throws Exception {
                    try {
                        LOGGER.info(format("%s Starting %s of %s %s", DASHES, testPhase.desc(), testId, DASHES));
                        test.invoke(testPhase);
                        LOGGER.info(format("%s Finished %s of %s %s", DASHES, testPhase.desc(), testId, DASHES));
                    } finally {
                        if (testPhase == TestPhase.LOCAL_TEARDOWN) {
                            tests.remove(testId);
                        }
                    }
                }
            };
            operationThread.start();
        } catch (Exception e) {
            LOGGER.fatal(format("Failed to execute %s of %s", testPhase.desc(), testId), e);
            throw e;
        }
    }

    private void processStartTest(StartTestOperation operation) {
        if (worker.startPerformanceMonitor()) {
            LOGGER.info(format("%s Starting performance monitoring %s", DASHES, DASHES));
        }

        final String testId = operation.getTestId();

        final TestContainer test = tests.get(testId);
        if (test == null) {
            LOGGER.warn(format("Failed to process operation %s (no test with testId %s is found)", operation, testId));
            return;
        }

        if (operation.isPassiveMember() && type == WorkerType.MEMBER) {
            LOGGER.info(format("%s Skipping run of %s (member is passive) %s", DASHES, testId, DASHES));
            sendPhaseCompletedOperation(testId, TestPhase.RUN);
            return;
        }

        OperationThread operationThread = new OperationThread(testId, TestPhase.RUN) {
            @Override
            public void doRun() throws Exception {
                LOGGER.info(format("%s Starting run of %s %s", DASHES, testId, DASHES));
                test.invoke(TestPhase.RUN);
                LOGGER.info(format("%s Completed run of %s %s", DASHES, testId, DASHES));

                // stop performance monitor if all tests have completed their run phase
                if (testsCompleted.incrementAndGet() == testsPending.get()) {
                    LOGGER.info(format("%s Stopping performance monitoring %s", DASHES, DASHES));
                    worker.shutdownPerformanceMonitor();
                }
            }
        };
        operationThread.start();
    }

    private void processStopTest(StopTestOperation operation) {
        String testId = operation.getTestId();
        TestContainer test = tests.get(testId);
        if (test == null) {
            LOGGER.warn("Can't stop test, found no test with id " + testId);
            return;
        }

        LOGGER.info(format("%s Stopping %s %s", DASHES, testId, DASHES));
        test.getTestContext().stop();
    }

    private void sendPhaseCompletedOperation(String testId, TestPhase testPhase) {
        PhaseCompletedOperation operation = new PhaseCompletedOperation(testId, testPhase);
        worker.getServerConnector().submit(COORDINATOR, operation);
    }

    private abstract class OperationThread extends Thread {

        private final String testId;

        public OperationThread(String testId, TestPhase testPhase) {
            this.testId = testId;

            TestPhase runningPhase = testPhases.putIfAbsent(testId, testPhase);
            if (runningPhase != null) {
                throw new IllegalStateException(format("Tried to start %s for test %s, but %s is still running!", testPhase,
                        testId, runningPhase));
            }
        }

        @Override
        public final void run() {
            try {
                doRun();
            } catch (Throwable t) {
                LOGGER.error("Error while executing test phase", t);
                exceptionLogger.log(t, testId);
            } finally {
                TestPhase testPhase = testPhases.remove(testId);
                sendPhaseCompletedOperation(testId, testPhase);
            }
        }

        public abstract void doRun() throws Exception;
    }
}
