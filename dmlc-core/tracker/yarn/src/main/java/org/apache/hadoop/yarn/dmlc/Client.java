package org.apache.hadoop.yarn.dmlc;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.QueueInfo;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;

import sun.misc.Signal;
import sun.misc.SignalHandler;

public class Client {
    // logger
    private static final Log LOG = LogFactory.getLog(Client.class);
    // permission for temp file
    private static final FsPermission permTemp = new FsPermission("777");
    // configuration
    private YarnConfiguration conf = new YarnConfiguration();
    // hdfs handler
    private FileSystem dfs;
    // cached maps
    private Map<String, String> cacheFiles = new java.util.HashMap<String, String>();
    // enviroment variable to setup cachefiles
    private String cacheFileArg = "";
    // args to pass to application master
    private String appArgs = "";
    // HDFS Path to store temporal result
    private String tempdir = "/tmp";
    // user name
    private String userName = "";
    // user credentials
    private Credentials credentials = null;
    // job name
    private String jobName = "";
    // queue
    private String queue = "default";
    // ApplicationMaster classpath
    private String appCp = null;
    // ApplicationMaster env
    private Map<String, String> env = new java.util.HashMap<String, String>();

    /**
     * constructor
     * @throws IOException
     */
    private Client() throws IOException {
        conf.addResource(new Path(System.getenv("HADOOP_CONF_DIR") +"/core-site.xml"));
        conf.addResource(new Path(System.getenv("HADOOP_CONF_DIR") +"/hdfs-site.xml"));
        dfs = FileSystem.get(conf);
        userName = UserGroupInformation.getCurrentUser().getShortUserName();
        credentials = UserGroupInformation.getCurrentUser().getCredentials();
    }

    /**
     * setup security token given current user
     * @return the ByeBuffer containing the security tokens
     * @throws IOException
     */
    private ByteBuffer setupTokens() throws IOException {
        DataOutputBuffer buffer = new DataOutputBuffer();
        String loc = System.getenv().get("HADOOP_TOKEN_FILE_LOCATION");
        if ((loc != null && loc.trim().length() > 0)
        ||  (!UserGroupInformation.isSecurityEnabled())) {
            this.credentials.writeTokenStorageToStream(buffer);
        } else {
            // Note: Credentials class is marked as LimitedPrivate for HDFS and MapReduce
            Credentials credentials = new Credentials();
            String tokenRenewer = conf.get(YarnConfiguration.RM_PRINCIPAL);
            if (tokenRenewer == null || tokenRenewer.length() == 0) {
                throw new IOException(
                "Can't get Master Kerberos principal for the RM to use as renewer");
            }

            // For now, only getting tokens for the default file-system.
            final Token<?> tokens[] = dfs.addDelegationTokens(tokenRenewer, credentials);
            if (tokens != null) {
                for (Token<?> token : tokens) {
                    LOG.info("Got dt for " + dfs.getUri() + "; " + token);
                }
            }
            credentials.writeTokenStorageToStream(buffer);
        }
        return ByteBuffer.wrap(buffer.getData(), 0, buffer.getLength());
    }

    /**
     * setup all the cached files
     *
     * @param fmaps
     *            the file maps
     * @return the resource map
     * @throws IOException
     */
    private Map<String, LocalResource> setupCacheFiles(ApplicationId appId) throws IOException {
        // create temporary dmlc directory
        Path tmpPath = new Path(this.tempdir);
        if (!dfs.exists(tmpPath)) {
            dfs.mkdirs(tmpPath, permTemp);
            LOG.info("HDFS temp directory do not exist, creating.. " + tmpPath);
        }
        tmpPath = new Path(tmpPath + "/temp-dmlc-yarn-" + appId);
        if (dfs.exists(tmpPath)) {
            dfs.delete(tmpPath, true);
        }
        // create temporary directory
        FileSystem.mkdirs(dfs, tmpPath, permTemp);

        StringBuilder cstr = new StringBuilder();
        Map<String, LocalResource> rmap = new java.util.HashMap<String, LocalResource>();
        for (Map.Entry<String, String> e : cacheFiles.entrySet()) {
            LocalResource r = Records.newRecord(LocalResource.class);
            Path path = new Path(e.getValue());
            // copy local data to temporary folder in HDFS
            if (!e.getValue().startsWith("hdfs://")) {
                Path dst = new Path("hdfs://" + tmpPath + "/"+  path.getName());
                dfs.copyFromLocalFile(false, true, path, dst);
                dfs.setPermission(dst, permTemp);
                dfs.deleteOnExit(dst);
                path = dst;
            }
            FileStatus status = dfs.getFileStatus(path);
            r.setResource(ConverterUtils.getYarnUrlFromPath(path));
            r.setSize(status.getLen());
            r.setTimestamp(status.getModificationTime());
            r.setType(LocalResourceType.FILE);
            r.setVisibility(LocalResourceVisibility.APPLICATION);
            rmap.put(e.getKey(), r);
            cstr.append(" -file \"");
            cstr.append(path.toString());
            cstr.append('#');
            cstr.append(e.getKey());
            cstr.append("\"");
        }

        dfs.deleteOnExit(tmpPath);
        this.cacheFileArg = cstr.toString();
        return rmap;
    }

    /**
     * get the environment variables for container
     *
     * @return the env variable for child class
     */
    private Map<String, String> getEnvironment() {
        // Setup environment variables

        if (appCp != null) {
            env.put("CLASSPATH", appCp);
        } else {
            StringBuilder cpath = new StringBuilder()
                .append(Environment.CLASSPATH.$$())
                .append(File.pathSeparatorChar)
                .append("." + File.pathSeparator + "*");
            for (String c : conf.getStrings(
                        YarnConfiguration.YARN_APPLICATION_CLASSPATH,
                        YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
                cpath.append(File.pathSeparatorChar)
                     .append(c.trim());
            }
            env.put("CLASSPATH", cpath.toString());
        }
        for (Map.Entry<String, String> e : System.getenv().entrySet()) {
            if (e.getKey().startsWith("DMLC_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey().startsWith("AWS_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey().startsWith("rabit_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey() == "LIBHDFS_OPTS") {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey().equals("LD_LIBRARY_PATH")) {
                env.put(e.getKey(), e.getValue());
            }
        }
        LOG.debug(env);
        return env;
    }

    /**
     * initialize the settings
     *
     * @param args
     */
    private void initArgs(String[] args) {
        // directly pass all arguments except args0
        StringBuilder sargs = new StringBuilder("");
        for (int i = 0; i < args.length; ++i) {
            if (args[i].equals("-file")) {
                String[] arr = args[++i].split("#");
                if (arr.length == 1) {
                    cacheFiles.put(new Path(arr[0]).getName(), arr[0]);
                } else {
                    cacheFiles.put(arr[1], arr[0]);
                }
            } else if(args[i].equals("-jobname")) {
                this.jobName = args[++i];
            } else if(args[i].equals("-tempdir")) {
                this.tempdir = args[++i];
            } else if(args[i].equals("-queue")) {
                this.queue = args[++i];
            } else if(args[i].equals("-appcp")) {
                this.appCp = args[++i];
            } else if(args[i].equals("-env")) {
                sargs.append(" ");
                sargs.append(args[i]);
                sargs.append(" ");
                sargs.append(args[i+1]);
                String[] pair = args[++i].split("=", 2);
                env.put(pair[0], (pair.length == 1) ? "" : pair[1]);
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
            System.out.println("options: [-file filename] [-appcp appClasspath]");
            return;
        }
        this.initArgs(args);
        // Create yarnClient
        YarnClient yarnClient = YarnClient.createYarnClient();
        yarnClient.init(conf);
        yarnClient.start();

        // Create application via yarnClient
        YarnClientApplication app = yarnClient.createApplication();

        // Set up the container launch context for the application master
        ContainerLaunchContext amContainer = Records
                .newRecord(ContainerLaunchContext.class);
        ApplicationSubmissionContext appContext = app
                .getApplicationSubmissionContext();
        // Submit application
        ApplicationId appId = appContext.getApplicationId();

        //add ctrl+c signal handler
        CtrlCHandler handler = new CtrlCHandler(appId, yarnClient);
        Signal intSignal = new Signal("INT");
        Signal.handle(intSignal, handler);

        // setup security token
        amContainer.setTokens(this.setupTokens());
        // setup cache-files and environment variables
        amContainer.setLocalResources(this.setupCacheFiles(appId));
        amContainer.setEnvironment(this.getEnvironment());
        String cmd = Environment.JAVA_HOME.$$() + "/bin/java"
                + " -Xmx900m"
                + " org.apache.hadoop.yarn.dmlc.ApplicationMaster"
                + this.cacheFileArg + ' ' + this.appArgs + " 1>"
                + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
                + " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr";

        LOG.debug(cmd);
        amContainer.setCommands(Collections.singletonList(cmd));

        // Set up resource type requirements for ApplicationMaster
        Resource capability = Records.newRecord(Resource.class);
        capability.setMemory(1024);
        capability.setVirtualCores(1);
        LOG.info("jobname=" + this.jobName + ",username=" + this.userName);

        appContext.setApplicationName(jobName + ":DMLC-YARN");
        appContext.setAMContainerSpec(amContainer);
        appContext.setResource(capability);
        appContext.setQueue(queue);
        //appContext.setUser(userName);
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
            System.out.println("Available queues:");
            for (QueueInfo q : yarnClient.getAllQueues()) {
              System.out.println(q.getQueueName());
            }

            yarnClient.killApplication(appId);
        }
    }

    class CtrlCHandler implements SignalHandler{
        private ApplicationId appId;
        private YarnClient yarnClient;
        public CtrlCHandler(ApplicationId appId, YarnClient yarnClient){
            this.appId = appId;
            this.yarnClient = yarnClient;
        }
        public void handle(Signal signal){
            try{
                yarnClient.killApplication(appId);
            }catch (Exception e){
                System.out.println("yarn client exception");
            }
        }
    }
    public static void main(String[] args) throws Exception {
        new Client().run(args);
    }
}
