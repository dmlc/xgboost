package org.apache.hadoop.yarn.dmlc;

import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;

/**
 * data structure to hold the task information
 */
public class TaskRecord {
    // task id of the task
    public int taskId = 0;
    // role of current node 
    public String taskRole = "worker";
    // number of failed attempts to run the task
    public int attemptCounter = 0;
    // container request, can be null if task is already running
    public ContainerRequest containerRequest = null;
    // running container, can be null if the task is not launched
    public Container container = null;
    // whether we have requested abortion of this task
    public boolean abortRequested = false;

    public TaskRecord(int taskId, String role) {
        this.taskId = taskId;
        this.taskRole = role;
    }
}
