# preprocess training datasets

sym = {'!', '?', '.', ',', ' ', '(', ')'}

def preprocess(path : str):
    output = open(f'{path[:-4]}-preprocessed.txt','w')
    count = 0
    with open(path,'r') as ds:
        for line in iter(ds.readline, ''):
            for i in line:
                if (i >= 'A' and i <= 'z') or (i in sym) or (i >= '0' and i <= '9'):
                    output.write(i)
                    count += 1
            if count >= 1e10:
                output.write('\n')
                count = 0


if __name__=='__main__':
    path = 'Stories/The_Last_Leaf.txt'
    preprocess(path)