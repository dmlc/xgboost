package org.apache.hadoop.yarn.rabit;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;

public class Client {
    // logger
    private static final Log LOG = LogFactory.getLog(ApplicationMaster.class);
    // configuration
    private YarnConfiguration conf = new YarnConfiguration();
    // cached maps
    private Map<String, Path> cacheFiles = new java.util.HashMap<String, Path>();
    // args to pass to application master
    private String appArgs;

    /**
     * get the local resource setting
     * 
     * @param fmaps
     *            the file maps
     * @return the resource map
     * @throws IOException
     */
    private Map<String, LocalResource> getLocalResource() throws IOException {
        Map<String, LocalResource> rmap = new java.util.HashMap<String, LocalResource>();
        for (Map.Entry<String, Path> e : cacheFiles.entrySet()) {
            LocalResource r = Records.newRecord(LocalResource.class);
            Path path = e.getValue();
            FileStatus status = FileSystem.get(conf).getFileStatus(path);
            r.setResource(ConverterUtils.getYarnUrlFromPath(path));
            r.setSize(status.getLen());
            r.setTimestamp(status.getModificationTime());
            r.setType(LocalResourceType.FILE);
            r.setVisibility(LocalResourceVisibility.PUBLIC);
            rmap.put(e.getKey(), r);
        }
        return rmap;
    }

    /**
     * get the environment variables for container
     * 
     * @return the env variable for child class
     */
    private Map<String, String> getEnvironment() {
        // Setup environment variables
        Map<String, String> env = new java.util.HashMap<String, String>();
        String cpath = "${CLASSPATH}:./*";
        for (String c : conf.getStrings(
                YarnConfiguration.YARN_APPLICATION_CLASSPATH,
                YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
            cpath += ':';
            cpath += c.trim();
        }
        env.put("CLASSPATH", cpath);
        for (Map.Entry<String, String> e : System.getenv().entrySet()) {
            if (e.getKey().startsWith("rabit_")) {
                env.put(e.getKey(), e.getValue());
            }
        }
        return env;
    }

    /**
     * initialize the settings
     * 
     * @param args
     */
    private void initArgs(String[] args) {
        // directly pass all args except args0
        StringBuilder sargs = new StringBuilder("");
        for (int i = 0; i < args.length; ++i) {
            if (args[i] == "-file") {
                String[] arr = args[i + 1].split("#");
                Path path = new Path(arr[0]);
                if (arr.length == 1) {
                    cacheFiles.put(path.getName(), path);
                } else {
                    cacheFiles.put(arr[1], path);
                }
                ++i;
            } else {
                sargs.append(" ");
                sargs.append(args[i]);
            }
        }
        this.appArgs = sargs.toString();
    }

    private void run(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Usage: [options] [commands..]");
            System.out.println("options: [-file filename]");
            return;
        }
        this.initArgs(args);
        // Create yarnClient
        YarnConfiguration conf = new YarnConfiguration();
        YarnClient yarnClient = YarnClient.createYarnClient();
        yarnClient.init(conf);
        yarnClient.start();

        // Create application via yarnClient
        YarnClientApplication app = yarnClient.createApplication();

        // Set up the container launch context for the application master
        ContainerLaunchContext amContainer = Records
                .newRecord(ContainerLaunchContext.class);
        amContainer.setCommands(Collections.singletonList("$JAVA_HOME/bin/java"
                + " -Xmx256M"
                + " org.apache.hadoop.yarn.rabit.ApplicationMaster"
                + this.appArgs + " 1>"
                + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
                + " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR
                + "/stderr"));

        // setup cache files
        amContainer.setLocalResources(this.getLocalResource());
        amContainer.setEnvironment(this.getEnvironment());

        // Set up resource type requirements for ApplicationMaster
        Resource capability = Records.newRecord(Resource.class);
        capability.setMemory(256);
        capability.setVirtualCores(1);

        ApplicationSubmissionContext appContext = app
                .getApplicationSubmissionContext();
        appContext.setApplicationName("Rabit-YARN");
        appContext.setAMContainerSpec(amContainer);
        appContext.setResource(capability);
        appContext.setQueue("default");

        // Submit application
        ApplicationId appId = appContext.getApplicationId();
        LOG.info("Submitting application " + appId);
        yarnClient.submitApplication(appContext);

        ApplicationReport appReport = yarnClient.getApplicationReport(appId);
        YarnApplicationState appState = appReport.getYarnApplicationState();
        while (appState != YarnApplicationState.FINISHED
                && appState != YarnApplicationState.KILLED
                && appState != YarnApplicationState.FAILED) {
            Thread.sleep(100);
            appReport = yarnClient.getApplicationReport(appId);
            appState = appReport.getYarnApplicationState();
        }

        System.out.println("Application " + appId + " finished with"
                + " state " + appState + " at " + appReport.getFinishTime());
        if (!appReport.getFinalApplicationStatus().equals(
                FinalApplicationStatus.SUCCEEDED)) {
            System.err.println(appReport.getDiagnostics());
        }
    }

    public static void main(String[] args) throws Exception {
        new Client().run(args);
    }
}
