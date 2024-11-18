import os
import pandas as pd

def get_data(root = "PrIMuS"):
    files_aa = {"png": [], "notation": []}
    files_ab = {"png": [], "notation": []}
    
    for pkg in os.listdir(root):
        try:
            for file in os.listdir(f"{root}/{pkg}"):
                path = f"{root}/{pkg}/{file}"
                with open(f"{path}/{file}.agnostic") as f:
                    data = f.read()
                
                if pkg == "package_aa":
                    files_aa["png"].append(f"{path}/{file}.png")
                    files_aa["notation"].append(data)
                    
                elif pkg == "package_ab":
                    files_ab["png"].append(f"{path}/{file}.png")
                    files_ab["notation"].append(data)
        except:
            break

    return files_aa, files_ab

def main():
    files_aa, files_ab = get_data()
    
    df_aa = pd.DataFrame(files_aa)
    df_aa.to_csv("package_aa.csv")

    df_ab = pd.DataFrame(files_ab)
    df_ab.to_csv("package_ab.csv")
    
if __name__ == "__main__":
    main()
