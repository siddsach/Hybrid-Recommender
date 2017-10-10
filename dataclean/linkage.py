#Record Linkage
#Link movie titles
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re
import time


def read_in(lens_file, scripts_file):
    '''
    Read in lens and scripts data.
    '''
    lens_df = pd.read_csv(lens_file)
    scripts_df = pd.read_csv(scripts_file)
    scripts_df = scripts_df.assign(movieId=-1)

    return lens_df, scripts_df


def basic_linkage(lens_df, scripts_df):
    scripts = scripts_df['title'].values.tolist()
    lens = lens_df['title'].values.tolist()
    exact = 0
    mismatched_fe = 0
    a_matches = 0
    an_matches = 0
    the_matches = 0
    start = time.time()

    for j, scripts_title in enumerate(scripts):
        found_match = False
        t0 = time.time()

        for i, lens_title in enumerate(lens):
            l = lens_title.lower().replace('&', 'and').replace(' ', '')
            s = scripts_title.lower().replace('&', 'and').replace(' ', '')
            l_np = re.sub(r'\W', '', l)
            s_np = re.sub(r'\W', '', s)

            # CASE 1 and 2
            if l_np == s_np:
                exact += 1
                found_match = True
                break
                
            else:
                l_year = l[-4:]
                s_year = s[-4:]

                # Case: Mismatched Foreign and English
                if l_year == s_year:
                    l_foreng = l.split('(')
                    s_foreng = s.split('(')

                    if len(l_foreng) == 3 and len(s_foreng) == 3:
                        l_combo = l_foreng[1]+l_foreng[0]+l_foreng[2]
                        l_stripped = re.sub(r'[^\w]','', l_combo.replace('&', 'and'))
                        s_combo = ''.join(s_foreng)
                        s_stripped = re.sub(r'[^\w]','', s_combo.replace('&', 'and'))

                        if l_stripped == s_stripped:
                            mismatched_fe += 1
                            found_match = True
                            break

                # Case: misplaced a, an, or the     
                    if ',a' in l:
                        if ''.join(l_np.split('a')) == ''.join(s_np.split('a')):
                            a_matches += 1
                            found_match = True
                            break
                    elif ',an' in l:
                        if ''.join(l_np.split('an')) == ''.join(s_np.split('an')):
                            an_matches += 1
                            found_match = True
                            break
                    elif ',the' in l:
                        if ''.join(l_np.split('the')) == ''.join(s_np.split('the')):
                            the_matches += 1
                            found_match = True
                            break
        if found_match:
            scripts_df.iat[j,2] = lens_df.iloc[i, 0]
            print(scripts_title)
            print(lens_title)
        print(time.time()-t0)
    
    scripts_df = scripts_df[scripts_df['movieId'] != -1].reset_index(drop=True)
    
    print('Exact matches: {}\nForeign-English: {}\n"a": {}\n"an": {}\n"the": {}'\
        .format(exact, mismatched_fe, a_matches, an_matches, the_matches))
    print("TIME: {}".format(time.time() - start))
    return scripts_df

if __name__ == '__main__':
    lens_file = "data/ml-20m/movies.csv"
    scripts_file = "data/scripts.csv"
    lens_df, scripts_df = read_in(lens_file, scripts_file)
    linked = basic_linkage(lens_df, scripts_df)
    linked.to_csv("data/matched.csv")
