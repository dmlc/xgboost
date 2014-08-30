using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kaggle_higgs_demo
{
    static internal class FormatXGBoost
    {

        static internal int test_from = -1;
        static internal int test_to = -1;
        static internal int test_length = 550000;
        static internal int training_length = 250000;
        static internal bool is_cv = false;
        static Dictionary<int, Event> training = new Dictionary<int, Event>();
        static internal double missing = -999.0;        

//---
        static internal void convert2xgboost(string training_path, string xgboost_training_path, string weight_training_path)
        {
            string first_line;
            using (StreamReader sr = new StreamReader(training_path))
            {
                first_line = sr.ReadLine();
                int line = 0;
                string curr_line;
                bool end_training = false;
                while (!end_training)
                {
                    curr_line = sr.ReadLine();
                    if (curr_line == null || curr_line.Trim().Length == 0)
                    {
                        end_training = true;
                    }
                    else
                    {
                        Event evt = new Event();
                        string[] fields = curr_line.Split(',');
                        for (int ifeat = 1; ifeat <= fields.Length -3; ifeat++)
			{
                            double par_val;
                            if (double.TryParse(fields[ifeat], out par_val))
	{
                                evt.features.Add(par_val);
		 
	} else {
        Console.WriteLine("Attention par_val not double: " + fields[ifeat]);
        evt.features.Add(missing);
        Console.WriteLine("Added missing !!!");
                            }
			}
                        //evt.features = string.Join(",", fields, 1, fields.Length - 3);
                        if (!double.TryParse(fields[fields.Length - 2], out evt.weight))
                        {
                            Console.WriteLine("wrong weight for line " + line.ToString());
                        }
                        evt.label = fields[fields.Length - 1];
                        if (evt.label.Equals("s"))
                        {
                            evt.label = "1";
                        }
                        if (evt.label.Equals("b"))
                        {
                            evt.label = "0";
                        }
                        training.Add(line++, evt);
                    }

                }
            }

            bool is_training = !(weight_training_path == null || weight_training_path.Trim().Length == 0);

            using (StreamWriter sw = new StreamWriter(xgboost_training_path, false))
            {
                for (int i = 0; i < training.Count; i++)
                {
                    if (is_cv)
                    {
                        if (is_training)
                        {
                            if (i >= test_from && i <= test_to)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            if (i < test_from || i > test_to)
                            {
                                continue;
                            }
                        }
                    }
                    
                    string curr_feature = "";
                    int ifeat = 0;
                    foreach (double par_val in training[i].features)
                    {
                        if (par_val != missing)
                        {
                            curr_feature += ifeat.ToString() + ":" + par_val.ToString(CultureInfo.InvariantCulture) + " ";
                        }
                        
                        ifeat++;
                        
                    }
                    sw.WriteLine(
                        training[i].label + ' '
                        + curr_feature.TrimEnd()
                        );
                    
                }
            }

            if (!is_training)
            {
                return;
            }

            using (StreamWriter sw = new StreamWriter(weight_training_path, false))
            {
                for (int i = 0; i < training.Count; i++)
                {
                    if (i>=test_from && i <= test_to)
	{
                                continue;
	}
                    sw.WriteLine(
                        (training[i].weight * (double)test_length / (double)training_length).ToString(CultureInfo.InvariantCulture)
                        );

                }
            }


        }
//---
    }
}
