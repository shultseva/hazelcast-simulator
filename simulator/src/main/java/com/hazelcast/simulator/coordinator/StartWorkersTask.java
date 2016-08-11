/*
 * Copyright (c) 2008-2016, Hazelcast, Inc. All Rights Reserved.
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
package com.hazelcast.simulator.coordinator;

import com.hazelcast.simulator.agent.workerprocess.WorkerProcessSettings;
import com.hazelcast.simulator.protocol.core.Response;
import com.hazelcast.simulator.protocol.core.ResponseType;
import com.hazelcast.simulator.protocol.core.SimulatorAddress;
import com.hazelcast.simulator.protocol.operation.CreateWorkerOperation;
import com.hazelcast.simulator.protocol.operation.StartTimeoutDetectionOperation;
import com.hazelcast.simulator.protocol.registry.ComponentRegistry;
import com.hazelcast.simulator.protocol.registry.WorkerData;
import com.hazelcast.simulator.utils.CommandLineExitException;
import com.hazelcast.simulator.utils.ThreadSpawner;
import org.apache.log4j.Logger;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static com.hazelcast.simulator.protocol.core.ResponseType.SUCCESS;
import static com.hazelcast.simulator.utils.CommonUtils.getElapsedSeconds;
import static com.hazelcast.simulator.utils.FormatUtils.HORIZONTAL_RULER;
import static java.lang.String.format;

/**
 * Starts all workers.
 * <p>
 * It receives a map with key the address of the agent to start workers on. And the value is a List of WorkerSettings; where
 * each item in this list corresponds to a single worker to create.
 * <p>
 * The workers will be created in order; first all member workers are started, and then all client workers are started. This
 * is done to prevent clients running into a non existing cluster.
 */
public class StartWorkersTask {
    private static final Logger LOGGER = Logger.getLogger(StartWorkersTask.class);

    private final RemoteClient remoteClient;
    private final ComponentRegistry componentRegistry;
    private final Echoer echoer;
    private final int startupDelayMs;
    private final Map<SimulatorAddress, List<WorkerProcessSettings>> memberDeploymentPlan;
    private final Map<SimulatorAddress, List<WorkerProcessSettings>> clientDeploymentPlan;
    private long started;

    public StartWorkersTask(
            Map<SimulatorAddress, List<WorkerProcessSettings>> deploymentPlan,
            RemoteClient remoteClient,
            ComponentRegistry componentRegistry,
            int startupDelayMs) {
        this.remoteClient = remoteClient;
        this.componentRegistry = componentRegistry;
        this.startupDelayMs = startupDelayMs;
        this.echoer = new Echoer(remoteClient);

        this.memberDeploymentPlan = filterByWorkerType(true, deploymentPlan);
        this.clientDeploymentPlan = filterByWorkerType(false, deploymentPlan);
    }

    public void run() {
        echoStartWorkers();

        // first create all members
        startWorkers(true, memberDeploymentPlan);
        // then create all clients
        startWorkers(false, clientDeploymentPlan);

        remoteClient.sendToAllAgents(new StartTimeoutDetectionOperation());
        remoteClient.startWorkerPingThread();

        if (componentRegistry.workerCount() > 0) {
            WorkerData firstWorker = componentRegistry.getFirstWorker();
            echoer.echo("Worker for global test phases will be %s (%s)", firstWorker.getAddress(),
                    firstWorker.getSettings().getWorkerType());
        }

        echoStartComplete();
    }

    private void echoStartWorkers() {
        started = System.nanoTime();
        echoer.echo(HORIZONTAL_RULER);
        echoer.echo("Starting Workers...");
        echoer.echo(HORIZONTAL_RULER);

        echoer.echo("Starting %d Workers (%d members, %d clients)...",
                count(memberDeploymentPlan) + count(clientDeploymentPlan),
                count(memberDeploymentPlan), count(clientDeploymentPlan));
    }

    private void echoStartComplete() {
        long elapsedSeconds = getElapsedSeconds(started);
        echoer.echo(HORIZONTAL_RULER);
        echoer.echo("Finished starting of %s Worker JVMs (%s seconds)",
                count(memberDeploymentPlan) + count(clientDeploymentPlan), elapsedSeconds);
        echoer.echo(HORIZONTAL_RULER);
    }

    private static Map<SimulatorAddress, List<WorkerProcessSettings>> filterByWorkerType(
            boolean isMember, Map<SimulatorAddress, List<WorkerProcessSettings>> deploymentPlan) {

        Map<SimulatorAddress, List<WorkerProcessSettings>> result = new HashMap<SimulatorAddress, List<WorkerProcessSettings>>();

        for (Map.Entry<SimulatorAddress, List<WorkerProcessSettings>> entry : deploymentPlan.entrySet()) {
            List<WorkerProcessSettings> filtered = new LinkedList<WorkerProcessSettings>();

            for (WorkerProcessSettings settings : entry.getValue()) {
                if (settings.getWorkerType().isMember() == isMember) {
                    filtered.add(settings);
                }
            }

            if (!filtered.isEmpty()) {
                result.put(entry.getKey(), filtered);
            }
        }
        return result;
    }

    private int count(Map<SimulatorAddress, List<WorkerProcessSettings>> deploymentPlan) {
        int result = 0;
        for (List<WorkerProcessSettings> settings : deploymentPlan.values()) {
            result += settings.size();
        }
        return result;
    }

    private void startWorkers(boolean isMember, Map<SimulatorAddress, List<WorkerProcessSettings>> deploymentPlan) {
        ThreadSpawner spawner = new ThreadSpawner("createWorkers", true);
        int workerIndex = 0;
        for (Map.Entry<SimulatorAddress, List<WorkerProcessSettings>> entry : deploymentPlan.entrySet()) {
            List<WorkerProcessSettings> workersSettings = entry.getValue();

            SimulatorAddress agentAddress = entry.getKey();
            String workerType = isMember ? "member" : "client";

            spawner.spawn(new StartWorkersOnAgentTask(workersSettings, startupDelayMs * workerIndex, agentAddress, workerType));

            if (isMember) {
                workerIndex++;
            }
        }
        spawner.awaitCompletion();
    }


    private final class StartWorkersOnAgentTask implements Runnable {
        private final List<WorkerProcessSettings> workersSettings;
        private final SimulatorAddress agentAddress;
        private final String workerType;
        private final int startupDelayMs;

        private StartWorkersOnAgentTask(List<WorkerProcessSettings> workersSettings,
                                        int startupDelaysMs,
                                        SimulatorAddress agentAddress,
                                        String workerType) {
            this.startupDelayMs = startupDelaysMs;
            this.workersSettings = workersSettings;
            this.agentAddress = agentAddress;
            this.workerType = workerType;
        }

        @Override
        public void run() {
            CreateWorkerOperation operation = new CreateWorkerOperation(workersSettings, startupDelayMs);
            Response response = remoteClient.getCoordinatorConnector().write(agentAddress, operation);

            ResponseType responseType = response.getFirstErrorResponseType();
            if (responseType != SUCCESS) {
                throw new CommandLineExitException(format("Could not create %d %s Worker on %s (%s)",
                        workersSettings.size(), workerType, agentAddress, responseType));
            }

            LOGGER.info(format("Created %d %s Worker on %s", workersSettings.size(), workerType, agentAddress));
            componentRegistry.addWorkers(agentAddress, workersSettings);
        }
    }
}
