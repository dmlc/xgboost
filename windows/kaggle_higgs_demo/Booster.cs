﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using xgboost_sharp_wrapper;

namespace kaggle_higgs_demo
{
    internal class Booster
    {
        internal IntPtr dtrain;
        internal IntPtr[] dmats;
        internal IntPtr boost;
        internal IntPtr dtest;

        internal Dictionary<int, Event> training_events = new Dictionary<int, Event>();
        internal Dictionary<int, Event> test_events = new Dictionary<int, Event>();
        internal Prediction[] array_pred;


        internal static Booster CreateBooster(xgboost xgb, string training_path, string test_path,
            Dictionary<int, Event> training_events, Dictionary<int, Event> test_events)
        {
            Booster booster = new Booster();
            booster.training_events = training_events;
            booster.test_events = test_events;
            booster.dtrain = xgb.SharpXGDMatrixCreateFromFile(training_path, 0);
            booster.dmats = new IntPtr[1];
            booster.dmats[0] = booster.dtrain;
            booster.boost = xgb.SharpXGBoosterCreate(booster.dmats, 1);
            booster.dtest = xgb.SharpXGDMatrixCreateFromFile(test_path, 0);
            return booster;
        }

        internal void Predict(xgboost xgb, double threshold_ratio, string test_path, Dictionary<int, Event> test_events)

        {
            this.test_events = test_events;
            this.dtest = xgb.SharpXGDMatrixCreateFromFile(test_path, 0);
            this.Predict(xgb, threshold_ratio);
        }

        internal void Predict(xgboost xgb, double threshold_ratio)
        {   
            float[] results = xgb.SharpXGBoosterPredict(boost, dtest, 0, (uint)test_events.Count);
            List<Prediction> preds = new List<Prediction>();

            int line = 0;
            foreach (double result in results)
            {
                Prediction curr_res = new Prediction();
                curr_res.id = test_events[line].id;
                curr_res.pred = result;
                preds.Add(curr_res);
                line++;
            }

            array_pred = preds.ToArray();

            Array.Sort(array_pred, delegate(Prediction pred1, Prediction pred2)
            {
                return pred1.pred.CompareTo(pred2.pred);
            });
            int norm_rank = 1;
            foreach (Prediction pred in array_pred)
            {
                pred.rank = norm_rank++;
                if ((array_pred.Length - pred.rank) / (double)array_pred.Length <= threshold_ratio)
                {
                    pred.is_signal = true;
                }
                else
                {
                    pred.is_signal = false;
                }
            }
            Array.Sort(array_pred, delegate(Prediction pred1, Prediction pred2)
            {
                return pred1.id.CompareTo(pred2.id);
            });
        }

        internal double compute_CV_AMS(xgboost xgb, int test_from, int test_to)
        {
            if ((test_to-test_from+1) != array_pred.Length)
            {
                Console.WriteLine("Wrong guesses number: " + array_pred.Length.ToString()
                    + " vs " + training_events.Count + " events");
                return -1;
            }
            double s_true = 0;
            double s_false = 0;
            double br = 10;
            double reset_s = 0;
            double reset_b = 0;
            for (int i = 0; i < array_pred.Length; i++)
            {
                if (training_events[test_from + i].label.Equals("s"))
                {
                    reset_s += training_events[test_from + i].weight;
                }
                else
                {
                    reset_b += training_events[test_from + i].weight;
                }
            }
            //weight statistics: wpos=691.989, wneg=411000, ratio=593.94

            for (int i = 0; i < array_pred.Length; i++)
            {
                if (training_events[test_from + i].label.Equals("s"))
                {
                    training_events[test_from + i].weight = training_events[test_from + i].weight * 691.989 / reset_s;
                }
                else
                {
                    training_events[test_from + i].weight = training_events[test_from + i].weight * 411000 / reset_b;
                }
            } 
            
            for (int i = 0; i < array_pred.Length; i++)
            {
                if (array_pred[i].is_signal)
                {
                    if (training_events[test_from+i].label.Equals("s"))
                    {
                        s_true += training_events[test_from + i].weight;
                    }
                    else
                    {
                        s_false += training_events[test_from + i].weight;
                    }
                }
            }
            double AMS = Math.Sqrt(
                    2 * (
                      (
                        (s_true + s_false + br) * Math.Log(1 + (s_true / (s_false + br)))
                      )
                      - s_true
                    )
                );
//weight statistics: wpos=691.989, wneg=411000, ratio=593.94
            
            Console.WriteLine("CV ams: " + AMS.ToString(CultureInfo.InvariantCulture) 
                //+" ratio from " + (reset_b / reset_s).ToString() + " to " + (411000.0 / 691.989).ToString()
                //+ " reset " + reset_s.ToString() + " to 691.989 and " + reset_b.ToString() + " to 411000 - "
                );
            return AMS;
        }

    }
}
