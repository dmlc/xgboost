import sys

def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]        
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print ("Usage: python trans_data.py [Ranksvm Format Input] [Output Feature File] [Output Group File]")
        sys.exit(0)

    fi = open(sys.argv[1])
    output_feature = open(sys.argv[2],"w")
    output_group = open(sys.argv[3],"w")
    
    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line:
            line = line[:line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            save_data(group_data,output_feature,output_group)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group)

    fi.close()
    output_feature.close()
    output_group.close()

