/*
 Copyright (c) 2014 by Contributors 

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
    
 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package org.dmlc.xgboost4j;

/**
 * interface for customized evaluation
 * 
 * @author hzx
 */
public interface IEvaluation {
	/**
	 * get evaluate metric
	 * 
	 * @return evalMetric
	 */
	public abstract String getMetric();

	/**
	 * evaluate with predicts and data
	 * 
	 * @param predicts
	 *            predictions as array
	 * @param dmat
	 *            data matrix to evaluate
	 * @return result of the metric
	 */
	public abstract float eval(float[][] predicts, DMatrix dmat);
}
