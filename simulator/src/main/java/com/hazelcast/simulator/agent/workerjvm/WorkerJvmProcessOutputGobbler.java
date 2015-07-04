package com.hazelcast.simulator.agent.workerjvm;

import com.hazelcast.util.EmptyStatement;
import edu.umd.cs.findbugs.annotations.SuppressFBWarnings;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;

import static com.hazelcast.simulator.utils.CommonUtils.closeQuietly;

@SuppressFBWarnings({"DM_DEFAULT_ENCODING"})
public class WorkerJvmProcessOutputGobbler extends Thread {

    private final PrintWriter writer;
    private final BufferedReader reader;

    public WorkerJvmProcessOutputGobbler(InputStream in, OutputStream out) {
        writer = new PrintWriter(out);
        reader = new BufferedReader(new InputStreamReader(in));
    }

    @Override
    public void run() {
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                writer.append(line).append("\n");
            }
        } catch (IOException ignored) {
            EmptyStatement.ignore(ignored);
        } finally {
            closeQuietly(writer);
            closeQuietly(reader);
        }
    }
}
