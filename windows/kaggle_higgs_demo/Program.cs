using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using xgboost_sharp_wrapper;
using System.IO;
using System.Globalization;

namespace kaggle_higgs_demo
{
    class Program
    {
        static void Main(string[] args)
        {
            string usage_str = "training_path.csv test_path.csv sharp_pred.csv";
            string[] usage_strings = usage_str.Split(' ');
            if (args.Length != usage_strings.Length)
            {
                Console.WriteLine("Usage: " + usage_str);
                Console.WriteLine("wrong number of arguments");
                return;
            }
            Dictionary<string, string> argsDictionary = new Dictionary<string, string>();
            for (int i = 0; i < args.Length; i++)
            {
                argsDictionary.Add(usage_strings[i], args[i]);
            }
            //try
            //{
                kaggle_higgs_demo(argsDictionary);
            //}
            /*catch (Exception exc)
            {
                Console.WriteLine(exc.Message);
            }*/
        }

        private static void kaggle_higgs_demo(Dictionary<string, string> argsDictionary)
        {
            xgboost xgb = new xgboost();
            Dictionary<int, Event> training_events;
            Dictionary<int, Event> training_events_cv1;
            Dictionary<int, Event> test_events;
            Dictionary<int, Event> test_events_cv1;
            string training_path = libsvm_format(argsDictionary["training_path.csv"], true, out training_events);
            string test_path = libsvm_format(argsDictionary["test_path.csv"], false, out test_events);
            Booster booster = Booster.CreateBooster(xgb, training_path, test_path, training_events, test_events);

            string training_path_cv1 = libsvm_format(argsDictionary["training_path.csv"], true, out training_events_cv1, true, 0, 49999);
            string test_path_cv1 = libsvm_format(argsDictionary["training_path.csv"], false, out test_events_cv1, true, 0, 49999);
            Booster booster_cv1 = Booster.CreateBooster(xgb, training_path_cv1, test_path_cv1, training_events_cv1, test_events_cv1);

            double threshold_ratio = 0.155;
            SetBoosterParameters(xgb, booster.boost, threshold_ratio);
            SetBoosterParameters(xgb, booster_cv1.boost, threshold_ratio);

            Console.WriteLine("training booster");
            int num_round = 155;
            for (int i = 0; i < num_round; i++)
            {
                xgb.SharpXGBoosterUpdateOneIter(booster.boost, i + 1, booster.dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(booster.boost, i + 1, booster.dmats, new string[2] { "train", "train" }, 1));
                Console.WriteLine("--- CV_1 starts ---");
                xgb.SharpXGBoosterUpdateOneIter(booster_cv1.boost, i + 1, booster_cv1.dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(booster_cv1.boost, i + 1, booster_cv1.dmats, new string[2] { "train", "train" }, 1));
                float[] results_cv1 = xgb.SharpXGBoosterPredict(booster_cv1.boost, booster_cv1.dtest, 0, 50000);
                booster_cv1.Predict(xgb, threshold_ratio);
                double ams = booster_cv1.compute_CV_AMS(xgb,0,49999);
                Console.WriteLine("--- CV_1 ends ---");
                Console.WriteLine("");
                
            }

            Console.WriteLine("loading training data end, start to predict csv");
            booster.Predict(xgb, threshold_ratio);
            
            
            using (StreamWriter sw = new StreamWriter(argsDictionary["sharp_pred.csv"], false))
            {

                sw.WriteLine("EventId, RankOrder, Class");
                foreach (Prediction pred in booster.array_pred)
                {
                    sw.WriteLine(pred.id.ToString() + "," + pred.rank.ToString() + "," + (pred.is_signal ? "s" : "b"));
                }
            }
            Console.WriteLine("submission ready");
        }

        

        private static void SetBoosterParameters(xgboost xgb, IntPtr boost, double threshold_ratio)
        {
            xgb.SharpXGBoosterSetParam(boost, "objective", "binary:logitraw");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "auc");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "ams@" + threshold_ratio.ToString(CultureInfo.InvariantCulture));
            //weight statistics: wpos=691.989, wneg=411000, ratio=593.94
            xgb.SharpXGBoosterSetParam(boost, "scale_pos_weight", "594.4");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "auc");
            xgb.SharpXGBoosterSetParam(boost, "silent", "0");
            xgb.SharpXGBoosterSetParam(boost, "eta", "0.10906");
            xgb.SharpXGBoosterSetParam(boost, "base_score", "0.52");
            xgb.SharpXGBoosterSetParam(boost, "max_depth", "9");
            xgb.SharpXGBoosterSetParam(boost, "nthread", "16");

        }



        private static string libsvm_format(string csv_path, bool has_weights, out Dictionary<int, Event> EventsDictionary,
            bool is_cv = false, int test_from = -1, int test_to = -1)
        {
            string libsvm_path;
            if (csv_path.EndsWith(".csv"))
            {
                libsvm_path = csv_path.Replace(".csv", ".libsvm");
            }
            else
            {
                libsvm_path = csv_path + ".libsvm";
            }
            if (is_cv)
            {
                libsvm_path += "."+test_from.ToString()+"-"+test_to.ToString();
                libsvm_path += ".cv";
                if (has_weights)
                {
                    libsvm_path += ".train";
                }
            }
            FormatXGBoost.EventsDictionary = new Dictionary<int, Event>();
            FormatXGBoost.is_cv = is_cv;
            FormatXGBoost.test_from = test_from;
            FormatXGBoost.test_to = test_to;
            if (has_weights)
            {
                FormatXGBoost.convert2xgboost(csv_path, libsvm_path, libsvm_path + ".weight");
            }
            else
            {
                FormatXGBoost.convert2xgboost(csv_path, libsvm_path, null);
            }
            EventsDictionary = new Dictionary<int, Event>();
            foreach (int key in FormatXGBoost.EventsDictionary.Keys)
            {
                /*if (is_cv)
                {
                    if (has_weights)
                    {
                        if (key >= test_from && key <= test_to)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if (key < test_from || key > test_to)
                        {
                            continue;
                        }
                    }
                }*/

                EventsDictionary.Add(key, FormatXGBoost.EventsDictionary[key]);
            }
            return libsvm_path;
        }


    }
}
