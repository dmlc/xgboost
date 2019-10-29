package ml.dmlc.xgboost4j.java;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class TrainTestData {
    static File getResourceFile(String resource) {
        InputStream is = TrainTestData.class.getResourceAsStream(resource);
        if (is == null) {
            throw new RuntimeException("Failed to resolve resource " + resource);
        }
        try {
            try {
                File file = File.createTempFile(resource.substring(1), "");
                Files.copy(is, file.toPath(), StandardCopyOption.REPLACE_EXISTING);
                file.deleteOnExit();
                return file;
            } finally {
                is.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load the resource " + resource, e);
        }
    }
}
