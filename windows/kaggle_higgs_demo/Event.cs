using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace kaggle_higgs_demo
{
    internal class Event
    {
        internal List<double> features = new List<double>();
        internal double weight;
        internal string label;
    }
}
