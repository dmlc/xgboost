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
            string usage_str = "training_path.libsvm test_path.libsvm sharp_pred.csv";
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
            IntPtr dtrain = xgb.SharpXGDMatrixCreateFromFile( (argsDictionary["training_path.libsvm"]), 0);
            IntPtr[] dmats = new IntPtr[1];
            dmats[0] = dtrain;
            Console.WriteLine("loading training data end, start to boost trees");
            IntPtr boost = xgb.SharpXGBoosterCreate(dmats,1);
            xgb.SharpXGBoosterSetParam(boost, "objective", "binary:logitraw");
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "auc");
            double threshold_ratio = 0.15;
            xgb.SharpXGBoosterSetParam(boost, "eval_metric", "ams@" + threshold_ratio.ToString(CultureInfo.InvariantCulture));
            //weight statistics: wpos=691.989, wneg=411000, ratio=593.94

            xgb.SharpXGBoosterSetParam(boost, "silent", "0");
            xgb.SharpXGBoosterSetParam(boost, "eta", "0.1");
            xgb.SharpXGBoosterSetParam(boost, "max_depth", "6");
            xgb.SharpXGBoosterSetParam(boost, "nthread", "16");

            Console.WriteLine("training booster");
            int num_round = 10;
            for (int i = 0; i < num_round; i++)
            {
                xgb.SharpXGBoosterUpdateOneIter(boost, i+1, dtrain);
                Console.WriteLine(xgb.SharpXGBoosterEvalOneIter(boost, i+1, dmats, new string[2] { "train","train" }, 1)); 
            }
            
            IntPtr dtest = xgb.SharpXGDMatrixCreateFromFile((argsDictionary["test_path.libsvm"]), 0);
            Console.WriteLine("loading training data end, start to predict csv");
            float[] results = xgb.SharpXGBoosterPredict(boost, dtest, 0, 550000);
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
            foreach(Prediction pred in array_pred)
            {
                pred.rank = norm_rank++;
                if ((array_pred.Length - pred.rank)/(double)array_pred.Length <= threshold_ratio)
                {
                    pred.is_signal = true;
                } else {
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
                    sw.WriteLine(pred.id.ToString() + "," + pred.rank.ToString() + "," + (pred.is_signal?"s":"b"));
                }
            }
            Console.WriteLine("submission ready");
        }

        /* not needed
         * static byte[] GetBytes(string str)
        {
            byte[] bytes = new byte[str.Length * sizeof(char)];
            System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }

        static string GetString(byte[] bytes)
        {
            char[] chars = new char[bytes.Length / sizeof(char)];
            System.Buffer.BlockCopy(bytes, 0, chars, 0, bytes.Length);
            return new string(chars);
        }*/
    }
}
