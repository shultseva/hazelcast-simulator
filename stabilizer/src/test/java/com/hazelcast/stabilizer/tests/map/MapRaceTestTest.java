package com.hazelcast.stabilizer.tests.map;

import com.hazelcast.stabilizer.tests.TestRunner;
import org.junit.Test;

public class MapRaceTestTest {

    @Test(expected = AssertionError.class)
    public void test() throws Throwable {
        MapRaceTest test = new MapRaceTest();
        new TestRunner(test).withDuration(10).run();
    }
}
