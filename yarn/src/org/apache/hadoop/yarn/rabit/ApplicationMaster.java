package org.apache.hadoop.yarn.rabit;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Collection;
import java.util.Collections;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.util.Records;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerExitStatus;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerState;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;

/**
 * application master for allocating resources of rabit client
 * 
 * @author Tianqi Chen
 */
public class ApplicationMaster {
	// logger 
	private static final Log LOG = LogFactory.getLog(ApplicationMaster.class);
	// configuration
	private Configuration conf = new YarnConfiguration();
	
	// number of cores allocated for each task
	private int numVCores = 1;
	// memory needed requested for the task
	private int numMemoryMB = 10;
	// priority of the app master
	private int appPriority = 0;
	// total number of tasks
	private int numTasks = 1;
	// maximum number of attempts to try in each task
	private int maxNumAttempt = 10;
	// command to launch
	private String command = "";
	
	// application tracker hostname
	private String appHostName = "";
	// tracker URL to do
	private String appTrackerUrl = "";
	// tracker port
	private int appTrackerPort = 0;

	// whether we start to abort the application, due to whatever fatal reasons
	private boolean startAbort = false;
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
	
    public static void main(String[] args) throws Exception {
    	new ApplicationMaster().run(args);
    }
    /**
     * get integer argument from environment variable
     * @param name name of key
     * @param required whether this is required
     * @param defv default value
     * @return the requested result
     */
    private int getEnvInteger(String name, boolean required, int defv) {
    	String value = System.getenv(name);
    	if (value == null) {
    		if (required) LOG.fatal("environment variable " + name + "not set");
    	}
    	return Integer.valueOf(value);
    }
    /**
     * initialize from arguments and command lines
     * @param args
     */
	private void initArgs(String args[]) {
		for (String c : args) {
			this.command += c + " ";
		}
		numVCores = this.getEnvInteger("rabit_cpu_vcores", true, numVCores);
		numMemoryMB = this.getEnvInteger("rabit_memory_mb", true, numMemoryMB);
		maxNumAttempt = this.getEnvInteger("rabit_max_attempt", false, maxNumAttempt);
	}
    /**
	 * called to start the application
	 */
	private void run(String args[]) throws Exception {
		this.initArgs(args);
		// list of tasks that waits to be submit 
		java.util.Collection<TaskRecord> tasks = new java.util.LinkedList<TaskRecord>();
		// add waiting tasks
		for (int i = 0; i < this.numTasks; ++i) {
			tasks.add(new TaskRecord(i));
		}
		this.rmClient = AMRMClientAsync.createAMRMClientAsync(1000, new RMCallbackHandler());
		this.nmClient = NMClientAsync.createNMClientAsync(new NMCallbackHandler());
		this.rmClient.init(conf);
		this.rmClient.start();
		this.nmClient.init(conf);
		this.nmClient.start();
		RegisterApplicationMasterResponse response =
			this.rmClient.registerApplicationMaster(this.appHostName, this.appTrackerPort, this.appTrackerUrl);
		Resource maxResource = response.getMaximumResourceCapability();
		if (maxResource.getMemory() < this.numMemoryMB) {
			LOG.warn("[Rabit] memory requested exceed bound " + maxResource.getMemory());
			this.numMemoryMB = maxResource.getMemory();
		}
		if (maxResource.getVirtualCores() < this.numVCores) {
			LOG.warn("[Rabit] memory requested exceed bound " + maxResource.getVirtualCores());
			this.numVCores = maxResource.getVirtualCores();
		}
		this.submitTasks(tasks);
		LOG.info("[Rabit] ApplicationMaster started");
		while (!this.doneAllJobs()) {
			try {
				Thread.sleep(100);;
			} catch (InterruptedException e) {
			}
	    }
		assert (killedTasks.size() + finishedTasks.size() == numTasks);
		boolean success = finishedTasks.size() == numTasks;	
	    LOG.info("Application completed. Stopping running containers");
	    nmClient.stop();
	    String diagnostics = "Diagnostics." + ", num_tasks" + this.numTasks
	            + ", finished=" + this.finishedTasks.size() + ", failed="
	            + this.killedTasks.size() + "\n" + this.abortDiagnosis;
		rmClient.unregisterApplicationMaster
			(success ? FinalApplicationStatus.SUCCEEDED : FinalApplicationStatus.FAILED,
			 diagnostics, appTrackerUrl);		
	}
	/**
	 * check if the job finishes
	 * @return whether we finished all the jobs
	 */
	private synchronized boolean doneAllJobs() {
		return pendingTasks.size() == 0 && runningTasks.size() == 0;
	}
	/**
	 * submit tasks to request containers for the tasks
	 * @param tasks a collection of tasks we want to ask container for 
	 */
	private synchronized void submitTasks(Collection<TaskRecord> tasks) {
		for (TaskRecord r : tasks) {
			Resource resource = Records.newRecord(Resource.class);
			resource.setMemory(numMemoryMB);
			resource.setVirtualCores(numVCores);
			Priority priority = Records.newRecord(Priority.class);
	        priority.setPriority(this.appPriority);
	        r.containerRequest = new ContainerRequest(resource, null, null, priority);
	        rmClient.addContainerRequest(r.containerRequest);
	        pendingTasks.add(r);
		}
	}
	/**
	 * launch the task on container
	 * @param container container to run the task
	 * @param task the task
	 */
	private void launchTask(Container container, TaskRecord task) {
		task.container = container;
		task.containerRequest = null;
        ContainerLaunchContext ctx =
        		Records.newRecord(ContainerLaunchContext.class);
        String cmd = command + " 1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout"
        		+ " 2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr";
        ctx.setCommands(Collections.singletonList(cmd));
        // setup environment variables
        Map<String, String> env = new java.util.HashMap<String, String>();
        // setup class path
	    StringBuilder cpath = new StringBuilder("${CLASSPATH}:./*");
	    for (String c : conf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH,
	    								YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
	    	cpath.append(':');
	    	cpath.append(c.trim());
	    }
	    env.put("CLASSPATH", cpath.toString());
	    // setup LD_LIBARY_pATH path for libhdfs
        env.put("LD_LIBRARY_PATH", "${LD_LIBRARY_PATH}:$HADOOP_HDFS_HOME/lib/native");
        // inherit all rabit variables
        for (Map.Entry<String, String> e : System.getenv().entrySet()) {
        	if (e.getKey().startsWith("rabit_")) {
            	env.put(e.getKey(), e.getValue());
            }
        }
        env.put("rabit_task_id", String.valueOf(task.taskId));
        env.put("rabit_num_trial", String.valueOf(task.attemptCounter));
        ctx.setEnvironment(env);
        synchronized (this) {
        	assert (!this.runningTasks.containsKey(container.getId()));
        	this.runningTasks.put(container.getId(), task);
        	this.nmClient.startContainerAsync(container, ctx);
        }
	}
	/**
	 * free the containers that have not yet been launched
	 * @param containers
	 */
	private synchronized void freeUnusedContainers(Collection<Container> containers) {
	}
	/**
	 * handle method for AMRMClientAsync.CallbackHandler container allocation
	 * @param containers
	 */
	private synchronized void onContainersAllocated(List<Container> containers) {
		if (this.startAbort) {
			this.freeUnusedContainers(containers);
			return;
		}
		Collection<Container> freelist = new java.util.LinkedList<Container>();
		for (Container c : containers) {
			TaskRecord task;
			task = pendingTasks.poll();
			if (task == null) {
				freelist.add(c); continue; 
			}
			this.launchTask(c, task);
		}
		this.freeUnusedContainers(freelist);
	}
	/**
	 * start aborting the job
	 * @param msg the fatal message
	 */
	private synchronized void abortJob(String msg) {
		if (!this.startAbort) this.abortDiagnosis = msg;
		this.startAbort = true;
		for (TaskRecord r : this.runningTasks.values()) {
			if (!r.abortRequested) {
				nmClient.stopContainerAsync(r.container.getId(), r.container.getNodeId());
				r.abortRequested = true;
			}
		}
		this.killedTasks.addAll(this.pendingTasks);
		for (TaskRecord r : this.pendingTasks) {
			rmClient.removeContainerRequest(r.containerRequest);
		}
		this.pendingTasks.clear();
		LOG.info(msg);
	}
	/**
	 * handle non fatal failures
	 * @param cid
	 */
	private synchronized void handleFailure(Collection<ContainerId> failed) {
		Collection<TaskRecord> tasks = new java.util.LinkedList<TaskRecord>();
		for (ContainerId cid : failed) {
			TaskRecord r = runningTasks.remove(cid);
			if (r  == null) continue;
			r.attemptCounter += 1;
			r.container = null;
			tasks.add(r);
			if (r.attemptCounter >= this.maxNumAttempt) {
				this.abortJob("[Rabit] Task " + r.taskId + " failed more than " + r.attemptCounter + "times");
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
	 * @param status list of status
	 */
	private synchronized void onContainersCompleted(List<ContainerStatus> status) {
		Collection<ContainerId> failed = new java.util.LinkedList<ContainerId>(); 
		for (ContainerStatus s : status) {
			assert (s.getState().equals(ContainerState.COMPLETE));
			int exstatus = s.getExitStatus();
			TaskRecord r = runningTasks.get(s.getContainerId());
			if (r == null) continue;
			if (exstatus == ContainerExitStatus.SUCCESS) {
				finishedTasks.add(r);
				runningTasks.remove(s.getContainerId());
			} else {
				switch (exstatus) {
				case ContainerExitStatus.KILLED_EXCEEDED_PMEM:
					this.abortJob("[Rabit] Task " + r.taskId + " killed because of exceeding allocated physical memory");
					break;
				case ContainerExitStatus.KILLED_EXCEEDED_VMEM:
					this.abortJob("[Rabit] Task " + r.taskId + " killed because of exceeding allocated virtual memory");
					break;
				default:
					LOG.info("[Rabit] Task " + r.taskId + " exited with status " + exstatus);
					failed.add(s.getContainerId());
				}
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
			return 1.0f - (float)(pendingTasks.size()) / numTasks;
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
			ApplicationMaster.this.abortJob("[Rabit] Resource manager Error " + ex.toString());
		}
		@Override
		public void onNodesUpdated(List<NodeReport> nodereport) {
		}
		@Override
		public void onShutdownRequest() {
			ApplicationMaster.this.abortJob("[Rabit] Get shutdown request, start to shutdown...");
		}
	}
	private class NMCallbackHandler implements NMClientAsync.CallbackHandler {
		@Override
		public void onContainerStarted(ContainerId cid, Map<String, ByteBuffer> services) {
			LOG.debug("onContainerStarted Invoked");
		}
		@Override
		public void onContainerStatusReceived(ContainerId cid, ContainerStatus status) {
			LOG.debug("onContainerStatusReceived Invoked");
		}
		@Override
		public void onContainerStopped(ContainerId cid) {
			LOG.debug("onContainerStopped Invoked");
		}
		@Override
		public void onGetContainerStatusError(ContainerId cid, Throwable ex) {
			LOG.debug("onGetContainerStatusError Invoked: " + ex.toString());
			ApplicationMaster.this.handleFailure(Collections.singletonList(cid));
		}
		@Override
		public void onStartContainerError(ContainerId cid, Throwable ex) {
			LOG.debug("onStartContainerError Invoked: " + ex.toString());
			ApplicationMaster.this.handleFailure(Collections.singletonList(cid));
		}
		@Override
		public void onStopContainerError(ContainerId cid, Throwable ex) {
			LOG.info("onStopContainerError Invoked: " + ex.toString());
		}
	}
}
