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
            try
            {
                kaggle_higgs_demo(argsDictionary);
            }
            catch (Exception exc)
            {
                Console.WriteLine(exc.Message);
            }
        }

        private static void kaggle_higgs_demo(Dictionary<string, string> argsDictionary)
        {
            xgboost xgb = new xgboost();
            string training_path = libsvm_format(argsDictionary["training_path.csv"], true);
            string test_path = libsvm_format(argsDictionary["test_path.csv"], false);
            
            Booster booster = CreateBooster(xgb, training_path, test_path);

            double threshold_ratio = 0.155;
            SetBoosterParameters(xgb, booster.boost, threshold_ratio);

            Console.WriteLine("training booster");
            int num_round = 155;
            for (int i = 0; i < num_round; i++)
            {
                xgb.SharpXGBoosterUpdateOneIter(booster.boost, i + 1, booster.dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(booster.boost, i + 1, booster.dmats, new string[2] { "train", "train" }, 1));
            }

            Console.WriteLine("loading training data end, start to predict csv");
            float[] results = xgb.SharpXGBoosterPredict(booster.boost, booster.dtest, 0, 550000);
            List<Prediction> preds = new List<Prediction>();

            int line = 0;
            foreach (double result in results)
            {
                Prediction curr_res = new Prediction();
                curr_res.id = 350000 + (++line);
                curr_res.pred = result;
                preds.Add(curr_res);
            }

            Prediction[] array_pred = preds.ToArray();
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
            using (StreamWriter sw = new StreamWriter(argsDictionary["sharp_pred.csv"], false))
            {

                sw.WriteLine("EventId, RankOrder, Class");
                foreach (Prediction pred in array_pred)
                {
                    sw.WriteLine(pred.id.ToString() + "," + pred.rank.ToString() + "," + (pred.is_signal ? "s" : "b"));
                }
            }
            Console.WriteLine("submission ready");
        }

        private static Booster CreateBooster(xgboost xgb, string training_path, string test_path)
        {
            Booster booster = new Booster();
            booster.dtrain = xgb.SharpXGDMatrixCreateFromFile(training_path, 0);
            booster.dmats = new IntPtr[1];
            booster.dmats[0] = booster.dtrain;
            Console.WriteLine("loading training data end, start to boost trees");
            booster.boost = xgb.SharpXGBoosterCreate(booster.dmats, 1);
            booster.dtest = xgb.SharpXGDMatrixCreateFromFile(test_path, 0);
            return booster;
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



        private static string libsvm_format(string csv_path, bool has_weights,
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
            return libsvm_path;
        }


    }
}
