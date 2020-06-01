import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        convert(new File("../../output_frames_pasta"), new File("../../output"), 0, 299, 60);
    }

    private static void convert(final File sourceDirectory, final File destinationDirectory, final int startFrameId, final int endFrameId, final int chunkSize){
        for (int i = startFrameId; i <= endFrameId; i += chunkSize){
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
            stdOutput.start("OUT", true);
            errorOutput.start("ERR", true);

            process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
