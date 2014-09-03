using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using xgboost_sharp_wrapper;
using System.IO;
using System.Globalization;
using System.Configuration;

namespace kaggle_higgs_demo
{
    class Program
    {
        static void Main(string[] args)
        {
            string usage_str = "training_path.csv test_path.csv sharp_pred.csv NFoldCV NRound";
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
            /*}
            catch (Exception exc)
            {
                Console.WriteLine(exc.Message);
            }*/
        }

        private static void kaggle_higgs_demo(Dictionary<string, string> argsDictionary)
        {
            xgboost xgb = new xgboost();
            Dictionary<int, Event> training_events;
            Dictionary<int, Event> Btraining_events;
            Dictionary<int, Event> test_events;
            string training_path = libsvm_format(550000, argsDictionary["training_path.csv"], true, out training_events);
            string Btraining_path = libsvm_format(550000, argsDictionary["training_path.csv"].Replace("1","2"), true, out Btraining_events);
            string test_path = libsvm_format(550000, argsDictionary["test_path.csv"], false, out test_events);
            
            Booster booster = Booster.CreateBooster(xgb, training_path, test_path, training_events, test_events);
            Booster Bbooster = Booster.CreateBooster(xgb, Btraining_path, test_path, Btraining_events, test_events);
            double threshold_ratio = 0.155;
            SetBoosterParameters(xgb, booster.boost, threshold_ratio);
            SetBoosterParameters(xgb, Bbooster.boost, threshold_ratio);
            //const int NFold = 10;
            int NFold;
            int.TryParse(argsDictionary["NFoldCV"], out NFold);
            int fold_test = NFold==0?0:training_events.Count / NFold;
            int fold_train = NFold == 0 ? 0 : training_events.Count - fold_test;
            Dictionary<int, Event>[] training_events_cv = NFold == 0 ? null : new Dictionary<int, Event>[NFold];
            Dictionary<int, Event>[] Btraining_events_cv = NFold == 0 ? null : new Dictionary<int, Event>[NFold];
            Dictionary<int, Event>[] test_events_cv = NFold == 0 ? null : new Dictionary<int, Event>[NFold];
            string[] training_path_cv = NFold == 0 ? null : new string[NFold];
            string[] Btraining_path_cv = NFold == 0 ? null : new string[NFold];
            string[] test_path_cv = NFold == 0 ? null : new string[NFold];
            Booster[] booster_cv = NFold == 0 ? null : new Booster[NFold];
            Booster[] Bbooster_cv = NFold == 0 ? null : new Booster[NFold];
            double[] ams_cv = NFold == 0 ? null : new double[NFold];
            for (int i = 0; i < NFold; i++)
            {
                training_path_cv[i] = libsvm_format(fold_test, argsDictionary["training_path.csv"], true, out training_events_cv[i], true, i * fold_test, ((i + 1) * fold_test) - 1);
                Btraining_path_cv[i] = libsvm_format(fold_test, argsDictionary["training_path.csv"].Replace("1","2"), true, out Btraining_events_cv[i], true, i * fold_test, ((i + 1) * fold_test) - 1);
                test_path_cv[i] = libsvm_format(fold_test, argsDictionary["training_path.csv"], false, out test_events_cv[i], true, i * fold_test, ((i + 1) * fold_test) - 1);
                booster_cv[i] = Booster.CreateBooster(xgb, training_path_cv[i], test_path_cv[i], training_events_cv[i], test_events_cv[i]);
                Bbooster_cv[i] = Booster.CreateBooster(xgb, Btraining_path_cv[i], test_path_cv[i], Btraining_events_cv[i], test_events_cv[i]);
                SetBoosterParameters(xgb, booster_cv[i].boost, threshold_ratio);
                SetBoosterParameters(xgb, Bbooster_cv[i].boost, threshold_ratio);
            };
            
            Console.WriteLine("training booster");

            int num_round;
            int.TryParse(argsDictionary["NRound"], out num_round);

            bool finish = false;
            double stop_ams;
            double.TryParse(ConfigurationManager.AppSettings["cv_ams"], out stop_ams);
            int curr_round = 0;
            while (!finish)
            //for (int i = 0; i < num_round; i++)
            {
                int i = curr_round;
                xgb.SharpXGBoosterUpdateOneIter(booster.boost, i + 1, booster.dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(booster.boost, i + 1, booster.dmats, new string[2] { "train", "train" }, 1));
                xgb.SharpXGBoosterUpdateOneIter(Bbooster.boost, i + 1, Bbooster.dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(Bbooster.boost, i + 1, Bbooster.dmats, new string[2] { "train", "train" }, 1));
                Parallel.For(0, NFold, j =>
                {
                    //Console.WriteLine("--- CV_" + j.ToString() + " starts ---");
                    xgb.SharpXGBoosterUpdateOneIter(booster_cv[j].boost, i + 1, booster_cv[j].dtrain);
                    xgb.SharpXGBoosterUpdateOneIter(Bbooster_cv[j].boost, i + 1, Bbooster_cv[j].dtrain);
                    //Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(booster_cv[j].boost, i + 1, booster_cv[j].dmats, new string[2] { "cv" + j.ToString(), "cv" + j.ToString() }, 1));
                    booster_cv[j].PredictRaw(xgb);
                    Bbooster_cv[j].PredictRaw(xgb);
                    Booster.MergePredict(threshold_ratio, booster_cv[j], Bbooster_cv[j]);
                    ams_cv[j] = Booster.Bcompute_CV_AMS(xgb, booster_cv[j], Bbooster_cv[j],j * fold_test, ((j + 1) * fold_test) - 1);
                    Console.Write("--- CV_" + j.ToString() + " ends");

                });
                if (NFold>0)
                {
                    double avg_ams = 0;
                    Console.WriteLine("\nround " + i.ToString() + ":");
;                    for (int j = 0; j < NFold; j++)
                    {
                        avg_ams += ams_cv[j];
                        Console.Write(" ams_cv" + j.ToString() + "=" + ams_cv[j].ToString(CultureInfo.InvariantCulture));
                    }
                    avg_ams = avg_ams / (double)NFold;
                    Console.WriteLine("\nround " + i.ToString() + ": AVG CV AMS = "
                        + avg_ams.ToString(CultureInfo.InvariantCulture));
                    Console.WriteLine("");
                    if (avg_ams >= stop_ams)
                    {
                        finish = true;
                    }
                }
                curr_round++;
                if (curr_round>num_round)
                {
                    finish = true;
                }
            }

            Console.WriteLine("loading training data end, start to predict csv");
            booster.PredictRaw(xgb);
            Bbooster.PredictRaw(xgb);
            Booster.MergePredict(threshold_ratio, booster, Bbooster);

            using (StreamWriter sw = new StreamWriter(argsDictionary["sharp_pred.csv"], false))
            {

                sw.WriteLine("EventId,RankOrder,Class");
                foreach (Prediction pred in booster.array_pred)
                {
                    sw.WriteLine(pred.id.ToString() + "," + pred.rank.ToString() + "," + (pred.is_signal ? "s" : "b"));
                }
            }
            Console.WriteLine("submission ready");

            for (int j = 0; j < NFold; j++)
            {
                booster_cv[j].PredictRaw(xgb, threshold_ratio, test_path, test_events);
                Bbooster_cv[j].PredictRaw(xgb, threshold_ratio, test_path, test_events);
                Booster.MergePredict(threshold_ratio, booster_cv[j],Bbooster_cv[j]);

                using (StreamWriter sw = new StreamWriter("cv"+j.ToString()+argsDictionary["sharp_pred.csv"], false))
                {

                    sw.WriteLine("EventId,RankOrder,Class");
                    foreach (Prediction pred in booster_cv[j].array_pred)
                    {
                        sw.WriteLine(pred.id.ToString() + "," + pred.rank.ToString() + "," + (pred.is_signal ? "s" : "b"));
                    }
                }
                Console.WriteLine("cv" + j.ToString() + " submission ready");
            }
        }

        

        private static void SetBoosterParameters(xgboost xgb, IntPtr boost, double threshold_ratio)
        {
            xgb.SharpXGBoosterSetParam(boost, "objective", ConfigurationManager.AppSettings["objective"]);  //binary:logitraw
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "auc");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "ams@" + threshold_ratio.ToString(CultureInfo.InvariantCulture));
            //weight statistics: wpos=691.989, wneg=411000, ratio=593.94
            xgb.SharpXGBoosterSetParam(boost, "scale_pos_weight", "594.4");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "auc");
            xgb.SharpXGBoosterSetParam(boost, "silent", "1");
            xgb.SharpXGBoosterSetParam(boost, "eta", ConfigurationManager.AppSettings["eta"]);
            xgb.SharpXGBoosterSetParam(boost, "base_score",  ConfigurationManager.AppSettings["base_score"]);
            xgb.SharpXGBoosterSetParam(boost, "max_depth", ConfigurationManager.AppSettings["max_depth"]);
            xgb.SharpXGBoosterSetParam(boost, "nthread", "8");

        }



        private static string libsvm_format(int test_length, string csv_path, bool has_weights, out Dictionary<int, Event> EventsDictionary,
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
            //FormatXGBoost.EventsDictionary = new Dictionary<int, Event>();
            //FormatXGBoost.is_cv = is_cv;
            //FormatXGBoost.test_from = test_from;
            //FormatXGBoost.test_to = test_to;
            if (has_weights)
            {
                FormatXGBoost.convert2xgboost(csv_path, libsvm_path, libsvm_path + ".weight", out EventsDictionary,
                    is_cv, test_from, test_to, test_length);
            }
            else
            {
                FormatXGBoost.convert2xgboost(csv_path, libsvm_path, null, out EventsDictionary,
                    is_cv, test_from, test_to, test_length);
            }
            /*EventsDictionary = new Dictionary<int, Event>();
            foreach (int key in FormatXGBoost.EventsDictionary.Keys)
            {

                EventsDictionary.Add(key, FormatXGBoost.EventsDictionary[key]);
            }*/
            return libsvm_path;
        }


    }
}
