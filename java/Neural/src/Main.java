import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        convert(new File("source"), new File("output"), 100, 50);
    }

    private static void convert(final File sourceDirectory, final File destinationDirectory, final int frameCount, final int chunkSize){
        for (int i = 0; i < frameCount; i += chunkSize){
            pythonConvert(sourceDirectory, destinationDirectory, i, i + chunkSize);
        }
    }

    private static void pythonConvert(final File sourceDirectory, final File destinationDirectory, final int startFrameId, final int endFrameId){
        final String[] command = new String[]{
                "python",
                "../../convert.py",
                sourceDirectory.getPath(),
                destinationDirectory.getPath(),
                String.valueOf(startFrameId),
                String.valueOf(endFrameId)
        };
        final ProcessBuilder processBuilder = new ProcessBuilder(command);
        System.out.println(processBuilder.command());

        try {
            final Process process = processBuilder.start();

            final StreamReader stdOutput = new StreamReader(process.getInputStream());
            final StreamReader errorOutput = new StreamReader(process.getErrorStream());
            stdOutput.start();
            errorOutput.start();

            process.waitFor();

            System.out.println(stdOutput.getOutput());
            System.out.println(errorOutput.getOutput());
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
