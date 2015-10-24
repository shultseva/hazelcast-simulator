package com.hazelcast.simulator.protocol.processors;

import com.hazelcast.simulator.coordinator.FailureContainer;
import com.hazelcast.simulator.coordinator.PerformanceStateContainer;
import com.hazelcast.simulator.coordinator.TestHistogramContainer;
import com.hazelcast.simulator.coordinator.TestPhaseListenerContainer;
import com.hazelcast.simulator.protocol.core.ResponseType;
import com.hazelcast.simulator.protocol.exception.LocalExceptionLogger;
import com.hazelcast.simulator.protocol.operation.IntegrationTestOperation;
import com.hazelcast.simulator.protocol.operation.SimulatorOperation;
import org.junit.Before;
import org.junit.Test;

import static com.hazelcast.simulator.protocol.core.ResponseType.UNSUPPORTED_OPERATION_ON_THIS_PROCESSOR;
import static com.hazelcast.simulator.protocol.core.SimulatorAddress.COORDINATOR;
import static com.hazelcast.simulator.protocol.operation.OperationType.getOperationType;
import static org.junit.Assert.assertEquals;

public class CoordinatorOperationProcessorTest {

    private CoordinatorOperationProcessor processor;

    @Before
    public void setUp() {
        LocalExceptionLogger exceptionLogger = new LocalExceptionLogger();
        TestPhaseListenerContainer testPhaseListenerContainer = new TestPhaseListenerContainer();
        PerformanceStateContainer performanceStateContainer = new PerformanceStateContainer();
        TestHistogramContainer testHistogramContainer = new TestHistogramContainer(performanceStateContainer);
        FailureContainer failureContainer = new FailureContainer("CoordinatorOperationProcessorTest", null);
        processor = new CoordinatorOperationProcessor(exceptionLogger, testPhaseListenerContainer, performanceStateContainer,
                testHistogramContainer, failureContainer);
    }

    @Test
    public void testProcessOperation_UnsupportedOperation() throws Exception {
        SimulatorOperation operation = new IntegrationTestOperation(IntegrationTestOperation.TEST_DATA);
        ResponseType responseType = processor.processOperation(getOperationType(operation), operation, COORDINATOR);

        assertEquals(UNSUPPORTED_OPERATION_ON_THIS_PROCESSOR, responseType);
    }
}
