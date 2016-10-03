package org.apache.hadoop.yarn.dmlc;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Collection;
import java.util.Collections;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerExitStatus;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerState;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;

/**
 * application master for allocating resources of dmlc client
 *
 * @author Tianqi Chen
 */
public class ApplicationMaster {
    // logger
    private static final Log LOG = LogFactory.getLog(ApplicationMaster.class);
    // configuration
    private Configuration conf = new YarnConfiguration();
    // hdfs handler
    private FileSystem dfs;

    // number of cores allocated for each server task
    private int workerCores = 1;
    // number of cores allocated for each worker task
    private int serverCores = 1;
    // memory needed requested for the worker task
    private int workerMemoryMB = 10;
    // memory needed requested for the server task
    private int serverMemoryMB = 10;
    // priority of the app master
    private int appPriority = 0;
    // total number of workers
    private int numWorker = 1;
    // total number of server
    private int numServer = 0;
    // total number of tasks
    private int numTasks;
    // maximum number of attempts to try in each task
    private int maxNumAttempt = 3;
    // command to launch
    private String command = "";

    // username
    private String userName = "";
    // user credentials
    private Credentials credentials = null;
    // application tracker hostname
    private String appHostName = "";
    // tracker URL to do
    private String appTrackerUrl = "";
    // tracker port
    private int appTrackerPort = 0;

    // whether we start to abort the application, due to whatever fatal reasons
    private boolean startAbort = false;
    // worker resources
    private Map<String, LocalResource> workerResources = new java.util.HashMap<String, LocalResource>();
    // record the aborting reason
    private String abortDiagnosis = "";
    // resource manager
    private AMRMClientAsync<ContainerRequest> rmClient = null;
    // node manager
    private NMClientAsync nmClient = null;

    // list of tasks that pending for resources to be allocated
    private final Queue<TaskRecord> pendingTasks = new java.util.LinkedList<TaskRecord>();
    // map containerId->task record of tasks that was running
    private final Map<ContainerId, TaskRecord> runningTasks = new java.util.HashMap<ContainerId, TaskRecord>();
    // collection of tasks
    private final Collection<TaskRecord> finishedTasks = new java.util.LinkedList<TaskRecord>();
    // collection of killed tasks
    private final Collection<TaskRecord> killedTasks = new java.util.LinkedList<TaskRecord>();
    // worker environment
    private final Map<String, String> env = new java.util.HashMap<String, String>();

    //add the blacklist
    private Collection<String> blackList = new java.util.HashSet();

    public static void main(String[] args) throws Exception {
        new ApplicationMaster().run(args);
    }

    private ApplicationMaster() throws IOException {
        dfs = FileSystem.get(conf);
        userName = UserGroupInformation.getCurrentUser().getShortUserName();
        credentials = UserGroupInformation.getCurrentUser().getCredentials();
    }


    /**
     * setup security token given current user
     * @return the ByeBuffer containing the security tokens
     * @throws IOException
     */
    private ByteBuffer setupTokens() {
        try {
            DataOutputBuffer dob = new DataOutputBuffer();
            credentials.writeTokenStorageToStream(dob);
            return ByteBuffer.wrap(dob.getData(), 0, dob.getLength()).duplicate();
        } catch (IOException e) {
            throw new RuntimeException(e);  // TODO: FIXME
        }
    }


    /**
     * get integer argument from environment variable
     *
     * @param name
     *            name of key
     * @param required
     *            whether this is required
     * @param defv
     *            default value
     * @return the requested result
     */
    private int getEnvInteger(String name, boolean required, int defv)
            throws IOException {
        String value = System.getenv(name);
        if (value == null) {
            if (required) {
                throw new IOException("environment variable " + name
                        + " not set");
            } else {
                return defv;
            }
        }
        return Integer.valueOf(value);
    }

    /**
     * initialize from arguments and command lines
     *
     * @param args
     */
    private void initArgs(String args[]) throws IOException {
        LOG.info("Start AM as user=" + this.userName);
        // get user name
        userName = UserGroupInformation.getCurrentUser().getShortUserName();
        // cached maps
        Map<String, Path> cacheFiles = new java.util.HashMap<String, Path>();
        for (int i = 0; i < args.length; ++i) {
            if (args[i].equals("-file")) {
                String[] arr = args[++i].split("#");
                Path path = new Path(arr[0]);
                if (arr.length == 1) {
                    cacheFiles.put(path.getName(), path);
                } else {
                    cacheFiles.put(arr[1], path);
                }
            } else if (args[i].equals("-env")) {
                String[] pair = args[++i].split("=", 2);
                env.put(pair[0], (pair.length == 1) ? "" : pair[1]);
            } else {
                this.command += args[i] + " ";
            }
        }
        for (Map.Entry<String, Path> e : cacheFiles.entrySet()) {
            LocalResource r = Records.newRecord(LocalResource.class);
            FileStatus status = dfs.getFileStatus(e.getValue());
            r.setResource(ConverterUtils.getYarnUrlFromPath(e.getValue()));
            r.setSize(status.getLen());
            r.setTimestamp(status.getModificationTime());
            r.setType(LocalResourceType.FILE);
            r.setVisibility(LocalResourceVisibility.APPLICATION);
            workerResources.put(e.getKey(), r);
        }
        workerCores = this.getEnvInteger("DMLC_WORKER_CORES", true, workerCores);
        serverCores = this.getEnvInteger("DMLC_SERVER_CORES", true, serverCores);
        workerMemoryMB = this.getEnvInteger("DMLC_WORKER_MEMORY_MB", true, workerMemoryMB);
        serverMemoryMB = this.getEnvInteger("DMLC_SERVER_MEMORY_MB", true, serverMemoryMB);
        numWorker = this.getEnvInteger("DMLC_NUM_WORKER", true, numWorker);
        numServer = this.getEnvInteger("DMLC_NUM_SERVER", true, numServer);
        numTasks = numWorker + numServer;
        maxNumAttempt = this.getEnvInteger("DMLC_MAX_ATTEMPT", false,
                                           maxNumAttempt);
        LOG.info("Try to start " + numServer + " Servers and " + numWorker + " Workers");
    }

    /**
     * called to start the application
     */
    private void run(String args[]) throws Exception {
        this.initArgs(args);
        this.rmClient = AMRMClientAsync.createAMRMClientAsync(1000,
                new RMCallbackHandler());
        this.nmClient = NMClientAsync
                .createNMClientAsync(new NMCallbackHandler());
        this.rmClient.init(conf);
        this.rmClient.start();
        this.nmClient.init(conf);
        this.nmClient.start();
        RegisterApplicationMasterResponse response = this.rmClient
                .registerApplicationMaster(this.appHostName,
                        this.appTrackerPort, this.appTrackerUrl);

        boolean success = false;
        String diagnostics = "";
        try {
            // list of tasks that waits to be submit
            java.util.Collection<TaskRecord> tasks = new java.util.LinkedList<TaskRecord>();
            // add waiting tasks
            for (int i = 0; i < this.numWorker; ++i) {
                tasks.add(new TaskRecord(i, "worker"));
            }
            for (int i = 0; i < this.numServer; ++i) {
                tasks.add(new TaskRecord(i, "server"));
            }
            Resource maxResource = response.getMaximumResourceCapability();

            if (maxResource.getMemory() < this.serverMemoryMB) {
              LOG.warn("[DMLC] memory requested exceed bound "
                        + maxResource.getMemory());
                this.serverMemoryMB = maxResource.getMemory();
            }
            if (maxResource.getMemory() < this.workerMemoryMB) {
              LOG.warn("[DMLC] memory requested exceed bound "
                        + maxResource.getMemory());
                this.workerMemoryMB = maxResource.getMemory();
            }
            if (maxResource.getVirtualCores() < this.workerCores) {
               LOG.warn("[DMLC] cores requested exceed bound "
                        + maxResource.getVirtualCores());
               this.workerCores = maxResource.getVirtualCores();
            }
            if (maxResource.getVirtualCores() < this.serverCores) {
              LOG.warn("[DMLC] cores requested exceed bound "
                        + maxResource.getVirtualCores());
                this.serverCores = maxResource.getVirtualCores();
            }
            this.submitTasks(tasks);
            LOG.info("[DMLC] ApplicationMaster started");
            while (!this.doneAllJobs()) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                }
            }
            assert (killedTasks.size() + finishedTasks.size() == numTasks);
            success = finishedTasks.size() == numTasks;
            LOG.info("Application completed. Stopping running containers");
            diagnostics = "Diagnostics." + ", num_tasks" + this.numTasks
                + ", finished=" + this.finishedTasks.size() + ", failed="
                + this.killedTasks.size() + "\n" + this.abortDiagnosis;
            nmClient.stop();
            LOG.info(diagnostics);
        } catch (Exception e) {
            diagnostics = e.toString();
        }
        rmClient.unregisterApplicationMaster(
                success ? FinalApplicationStatus.SUCCEEDED
                        : FinalApplicationStatus.FAILED, diagnostics,
                appTrackerUrl);
        if (!success)
            throw new Exception("Application not successful");
    }

    /**
     * check if the job finishes
     *
     * @return whether we finished all the jobs
     */
    private synchronized boolean doneAllJobs() {
        return pendingTasks.size() == 0 && runningTasks.size() == 0;
    }

    /**
     * submit tasks to request containers for the tasks
     *
     * @param tasks
     *            a collection of tasks we want to ask container for
     */
    private synchronized void submitTasks(Collection<TaskRecord> tasks) {
        for (TaskRecord r : tasks) {
            Resource resource = Records.newRecord(Resource.class);
            if (r.taskRole == "server") {
              resource.setMemory(serverMemoryMB);
              resource.setVirtualCores(serverCores);
            } else {
              resource.setMemory(workerMemoryMB);
              resource.setVirtualCores(workerCores);
            }
            Priority priority = Records.newRecord(Priority.class);
            priority.setPriority(this.appPriority);
            r.containerRequest = new ContainerRequest(resource, null, null,
                    priority);
            rmClient.addContainerRequest(r.containerRequest);
            pendingTasks.add(r);
        }
    }



    private synchronized void launchDummyTask(Container container){
        ContainerLaunchContext ctx = Records.newRecord(ContainerLaunchContext.class);
        String new_command = "./launcher.py";
        String cmd = new_command + " 1>"
            + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
            + " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR
            + "/stderr";
        ctx.setCommands(Collections.singletonList(cmd));
        ctx.setTokens(setupTokens());
        ctx.setLocalResources(this.workerResources);
        synchronized (this){
            this.nmClient.startContainerAsync(container, ctx);
        }
    }
    /**
     * launch the task on container
     *
     * @param container
     *            container to run the task
     * @param task
     *            the task
     */
    private void launchTask(Container container, TaskRecord task) {
        task.container = container;
        task.containerRequest = null;
        ContainerLaunchContext ctx = Records
                .newRecord(ContainerLaunchContext.class);
        String cmd =
        // use this to setup CLASSPATH correctly for libhdfs
             this.command + " 1>"
            + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
            + " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR
            + "/stderr";
        ctx.setCommands(Collections.singletonList(cmd));
        // TODO: token was not right
        ctx.setTokens(setupTokens());
        LOG.info(workerResources);
        ctx.setLocalResources(this.workerResources);
        // setup environment variables

        boolean isWindows = System.getProperty("os.name").startsWith("Windows");
        // setup class path, this is kind of duplicated, ignoring
        String classPathStr = isWindows? "%CLASSPATH%" : "${CLASSPATH}";
        StringBuilder cpath = new StringBuilder(classPathStr
                       + File.pathSeparatorChar
                       + "./*");
        for (String c : conf.getStrings(
                YarnConfiguration.YARN_APPLICATION_CLASSPATH,
                YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
            if (isWindows) c = c.replace('\\', '/');
            String[] arrPath = c.split("" + File.pathSeparatorChar);
            for (String ps : arrPath) {
                if (ps.endsWith("*.jar")
                        || ps.endsWith("*")
                        || ps.endsWith("/")) {
                    ps = ps.substring(0, ps.lastIndexOf('*'));
                    if (ps.startsWith("$") || ps.startsWith("%")) {
                        String[] arr =ps.split("/", 2);
                        if (arr.length != 2) continue;
                        try {
                            String vname = isWindows ?
                                           arr[0].substring(1, arr[0].length() - 1) :
                                           arr[0].substring(1);
                            String vv = System.getenv(vname);
                            if (isWindows) vv = vv.replace('\\', '/');
                                ps = vv + '/' + arr[1];
                        } catch (Exception e){
                            continue;
                        }
                    }
                    File dir = new File(ps);
                    if (dir.isDirectory()) {
                        for (File f: dir.listFiles()) {
                            if (f.isFile() && f.getPath().endsWith(".jar")) {
                                cpath.append(File.pathSeparatorChar);
                                cpath.append(ps + '/' + f.getName());
                            }
                        }
                    }
                    cpath.append(File.pathSeparatorChar);
                    cpath.append(ps + '/');
                } else {
                    cpath.append(File.pathSeparatorChar);
                    cpath.append(ps.trim());
                }
            }
        }
        // already use hadoop command to get class path in worker, maybe a
        // better solution in future
        env.put("CLASSPATH", cpath.toString());
        // setup LD_LIBARY_PATH path for libhdfs
        String oldLD_LIBRARY_PATH = System.getenv("LD_LIBRARY_PATH");
        env.put("LD_LIBRARY_PATH",
                oldLD_LIBRARY_PATH == null ? "" : oldLD_LIBRARY_PATH + ":$HADOOP_HDFS_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server");
        env.put("PYTHONPATH", "${PYTHONPATH}:.");
        // inherit all rabit variables
        for (Map.Entry<String, String> e : System.getenv().entrySet()) {
            if (e.getKey().startsWith("DMLC_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey().startsWith("rabit_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey().startsWith("AWS_")) {
                env.put(e.getKey(), e.getValue());
            }
            if (e.getKey() == "LIBHDFS_OPTS") {
                env.put(e.getKey(), e.getValue());
            }
        }
        String nodeHost = container.getNodeId().getHost();
        env.put("DMLC_NODE_HOST", nodeHost);
        env.put("DMLC_TASK_ID", String.valueOf(task.taskId));
        env.put("DMLC_ROLE", task.taskRole);
        env.put("DMLC_NUM_ATTEMPT", String.valueOf(task.attemptCounter));
        // ctx.setUser(userName);
        ctx.setEnvironment(env);
        LOG.info(env);
        synchronized (this) {
            assert (!this.runningTasks.containsKey(container.getId()));
            this.runningTasks.put(container.getId(), task);
            this.nmClient.startContainerAsync(container, ctx);
        }
    }
    /**
     * free the containers that have not yet been launched
     *
     * @param containers
     */
    private synchronized void onStartContainerError(ContainerId cid) {
        ApplicationMaster.this.handleFailure(Collections.singletonList(cid));
    }
    /**
     * free the containers that have not yet been launched
     *
     * @param containers
     */
    private synchronized void freeUnusedContainers(
            Collection<Container> containers) {
        if(containers.size() == 0) return;
        for(Container c : containers){
            launchDummyTask(c);
        }
    }

    /**
     * handle method for AMRMClientAsync.CallbackHandler container allocation
     *
     * @param containers
     */
    private synchronized void onContainersAllocated(List<Container> containers) {
        if (this.startAbort) {
            this.freeUnusedContainers(containers);
            return;
        }
        Collection<Container> freelist = new java.util.LinkedList<Container>();
        for (Container c : containers) {
            if(blackList.contains(c.getNodeHttpAddress())){
			    launchDummyTask(c);
                continue;
		    }

            TaskRecord task;
            task = pendingTasks.poll();
            if (task == null) {
                freelist.add(c);
                continue;
            }
            this.launchTask(c, task);
        }
        this.freeUnusedContainers(freelist);
    }

    /**
     * start aborting the job
     *
     * @param msg
     *            the fatal message
     */
    private synchronized void abortJob(String msg) {
        if (!this.startAbort)
            this.abortDiagnosis = msg;
        this.startAbort = true;
        for (TaskRecord r : this.runningTasks.values()) {
            if (!r.abortRequested) {
                nmClient.stopContainerAsync(r.container.getId(),
                        r.container.getNodeId());
                r.abortRequested = true;

                this.killedTasks.add(r);
            }
        }
        this.killedTasks.addAll(this.pendingTasks);
        for (TaskRecord r : this.pendingTasks) {
            rmClient.removeContainerRequest(r.containerRequest);
        }
        this.pendingTasks.clear();
        this.runningTasks.clear();
        LOG.info(msg);
    }

    /**
     * handle non fatal failures
     *
     * @param cid
     */
    private synchronized void handleFailure(Collection<ContainerId> failed) {
        Collection<TaskRecord> tasks = new java.util.LinkedList<TaskRecord>();
        for (ContainerId cid : failed) {
            TaskRecord r = runningTasks.remove(cid);
            if (r == null) {
                continue;
            }
            LOG.info("Task "
                    + r.taskId
                    + " failed on "
                    + r.container.getId()
                    + ". See LOG at : "
                    + String.format("http://%s/node/containerlogs/%s/"
                            + userName, r.container.getNodeHttpAddress(),
                            r.container.getId()));
            r.attemptCounter += 1;

            //stop the failed container and add it to blacklist
            nmClient.stopContainerAsync(r.container.getId(), r.container.getNodeId());
            blackList.add(r.container.getNodeHttpAddress());

            r.container = null;
            tasks.add(r);
            if (r.attemptCounter >= this.maxNumAttempt) {
                this.abortJob("[DMLC] Task " + r.taskId + " failed more than "
                        + r.attemptCounter + "times");
            }
        }
        if (this.startAbort) {
            this.killedTasks.addAll(tasks);
        } else {
            this.submitTasks(tasks);
        }
    }

    /**
     * handle method for AMRMClientAsync.CallbackHandler container allocation
     *
     * @param status
     *            list of status
     */
    private synchronized void onContainersCompleted(List<ContainerStatus> status) {
        Collection<ContainerId> failed = new java.util.LinkedList<ContainerId>();
        for (ContainerStatus s : status) {
            assert (s.getState().equals(ContainerState.COMPLETE));
            int exstatus = s.getExitStatus();
            TaskRecord r = runningTasks.get(s.getContainerId());
            if (r == null)
                continue;
            if (exstatus == ContainerExitStatus.SUCCESS) {
                finishedTasks.add(r);
                runningTasks.remove(s.getContainerId());
            } else {
                try {
                    if (exstatus == ContainerExitStatus.class.getField(
                            "KILLED_EXCEEDED_PMEM").getInt(null)) {
                        this.abortJob("[DMLC] Task "
                                + r.taskId
                                + " killed because of exceeding allocated physical memory");
                        return;
                    }
                    if (exstatus == ContainerExitStatus.class.getField(
                            "KILLED_EXCEEDED_VMEM").getInt(null)) {
                        this.abortJob("[DMLC] Task "
                                + r.taskId
                                + " killed because of exceeding allocated virtual memory");
                        return;
                    }
                } catch (Exception e) {
                    LOG.warn(e.getMessage());
                }
                LOG.info("[DMLC] Task " + r.taskId + " exited with status "
                         + exstatus + " Diagnostics:"+ s.getDiagnostics());
                failed.add(s.getContainerId());
            }
        }
        this.handleFailure(failed);
    }

    /**
     * callback handler for resource manager
     */
    private class RMCallbackHandler implements AMRMClientAsync.CallbackHandler {
        @Override
        public float getProgress() {
            return 1.0f - (float) (pendingTasks.size()) / numTasks;
        }

        @Override
        public void onContainersAllocated(List<Container> containers) {
            ApplicationMaster.this.onContainersAllocated(containers);
        }

        @Override
        public void onContainersCompleted(List<ContainerStatus> status) {
            ApplicationMaster.this.onContainersCompleted(status);
        }

        @Override
        public void onError(Throwable ex) {
            ApplicationMaster.this.abortJob("[DMLC] Resource manager Error "
                    + ex.toString());
        }

        @Override
        public void onNodesUpdated(List<NodeReport> nodereport) {
        }

        @Override
        public void onShutdownRequest() {
            ApplicationMaster.this
                    .abortJob("[DMLC] Get shutdown request, start to shutdown...");
        }
    }

    private class NMCallbackHandler implements NMClientAsync.CallbackHandler {
        @Override
        public void onContainerStarted(ContainerId cid,
                Map<String, ByteBuffer> services) {
            LOG.info("onContainerStarted Invoked");
        }

        @Override
        public void onContainerStatusReceived(ContainerId cid,
                ContainerStatus status) {
            LOG.info("onContainerStatusReceived Invoked");
        }

        @Override
        public void onContainerStopped(ContainerId cid) {
            LOG.info("onContainerStopped Invoked");
        }

        @Override
        public void onGetContainerStatusError(ContainerId cid, Throwable ex) {
            LOG.info("onGetContainerStatusError Invoked: " + ex.toString());
            ApplicationMaster.this
                    .handleFailure(Collections.singletonList(cid));
        }

        @Override
        public void onStartContainerError(ContainerId cid, Throwable ex) {
            LOG.info("onStartContainerError Invoked: " + ex.getMessage());
            ApplicationMaster.this
               .onStartContainerError(cid);
        }

        @Override
        public void onStopContainerError(ContainerId cid, Throwable ex) {
            LOG.info("onStopContainerError Invoked: " + ex.toString());
        }
    }
}
