import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

public class StreamReader {

    /**
     * {@link InputStream} to read from.
     */
    private final InputStream inputStream;

    /**
     * {@link StringBuilder} to append data to.
     */
    private final StringBuilder stringBuilder = new StringBuilder();

    private volatile boolean running = false;

    /**
     * Create a new StreamReader with the given input stream. To start the reader, call {@link StreamReader#start()}.
     * @param inputStream
     */
    public StreamReader(final InputStream inputStream){
        this.inputStream = inputStream;
    }

    /**
     * Start reading the stream in a new thread. This will keep reading until the end of the stream, or until
     * {@link StreamReader#stop()} is called.
     */
    public void start(final String tag, final boolean live){
        running = true;
        new Thread(() -> {
            try {
                final BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while (running && (line = bufferedReader.readLine()) != null){
                    if (live) System.out.println("[" + tag + "] " + line);
                    stringBuilder.append(line);
                    stringBuilder.append('\n');
                }
            }catch (Exception e){
                e.printStackTrace();
            }
            running = false;
        }).start();
    }

    /**
     * Wait until the end of the stream has been reached. This will pause the current thread until the end of the stream
     * has been reached. This could potentially be forever.
     */
    public void waitFor(){
        while (running);
    }

    public String getOutput(){
        return stringBuilder.toString();
    }

    public void stop(){
        running = false;
    }
}
