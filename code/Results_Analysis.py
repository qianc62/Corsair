import _public as pb
import numpy as np
import re
import math


def Extract(string, keyword, index=1):
    objs = string.split(keyword)
    num = ''
    for ch in objs[index]:
        if ch in ['0','1','2','3','4','5','6','7','8','9','.','-','e']:
            num += ch
        else:
            if ch=='%':
                return float(num)/100.0
            else:
                if 'e' not in num:
                    return float(num)
                objs = num.split('e')
                return float(objs[0])*math.pow(10, int(objs[1]))

Instance_IterationNum_Control = 10

instances, configurations = {}, {}
# epochs->instance
lines = open(pb.Save_Path).read().split('\n')
for i,line in enumerate(lines):
    if 'counterfactual_label_fairness' in line:
        objs = [obj.strip() for obj in re.split(r'([|])', line) if obj not in ['|'] and len(obj.strip())>0]
        instance_name = '-'.join(objs[0].split('-')[:4])
        epoch = int(objs[0].split('-')[5].split('=')[-1])

        dev_factual_maf1 = Extract(line, 'dev_factual_maf1=')
        dev_counterfactual_maf1 = Extract(line, 'dev_counterfactual_maf1=')

        test_factual_maf1 = Extract(line, 'test_counterfactual_maf1=', index=2)
        test_counterfactual_maf1 = Extract(line, 'test_counterfactual_maf1=', index=5)

        factual_label_fairness = Extract(line, 'factual_label_fairness=')
        counterfactual_label_fairness = Extract(line, 'counterfactual_label_fairness=')

        factual_keyword_fairness = Extract(line, 'factual_keyword_fairness=')
        counterfactual_keyword_fairness = Extract(line, 'counterfactual_keyword_fairness=')

        rate = line.split('(rate=')[1].split(')')[0]

        if instance_name not in instances.keys():
            instances[instance_name] = []
        instances[instance_name].append([dev_factual_maf1, dev_counterfactual_maf1,
                                         test_factual_maf1, test_counterfactual_maf1,
                                         factual_label_fairness, counterfactual_label_fairness,
                                         factual_keyword_fairness, counterfactual_keyword_fairness,
                                         rate])

# instances->configuration
for instance_name in instances:
    if len(instances[instance_name])<Instance_IterationNum_Control:
        pb.Print('{} {} {}'.format(instance_name, len(instances[instance_name]), 'epoch not enough'), color='red')
        continue
    # print(instance_name)
    fdev_maf1s = np.array([result[0] for result in instances[instance_name]])
    cdev_maf1s = np.array([result[1] for result in instances[instance_name]])
    ftest_maf1s = np.array([result[2] for result in instances[instance_name]])
    ctest_maf1s = np.array([result[3] for result in instances[instance_name]])
    ftest_lfs = np.array([result[4] for result in instances[instance_name]])
    ctest_lfs = np.array([result[5] for result in instances[instance_name]])
    ftest_kfs = np.array([result[6] for result in instances[instance_name]])
    ctest_kfs = np.array([result[7] for result in instances[instance_name]])
    rates = np.array([result[8] for result in instances[instance_name]])

    configuration_name = instance_name.split('-Seed')[0]
    if configuration_name not in configurations.keys():
        configurations[configuration_name] = []

    result = []
    index = np.argmax(cdev_maf1s)
    result.append(ftest_maf1s[index])
    result.append(ctest_maf1s[index])
    result.append(ftest_lfs[index])
    result.append(ctest_lfs[index])
    result.append(ftest_kfs[index])
    result.append(ctest_kfs[index])
    result.append(rates[index])
    configurations[configuration_name].append(result)

def Statistic(array):
    return np.average(array), np.std(array)

def Final_Statistic(model_name):
    factual_ans = []
    counterfactual_ans = []
    p_ans = []

    # output average results
    final_map = {}
    for configuration_name in configurations.keys():
        if model_name not in configuration_name: continue

        results = configurations[configuration_name]
        ff1s = [result[0] for result in results]
        cf1s = [result[1] for result in results]
        flfs = [result[2] for result in results]
        clfs = [result[3] for result in results]
        fkfs = [result[4] for result in results]
        ckfs = [result[5] for result in results]
        rates = [result[6] for result in results]

        ff1, ff1_std = Statistic(ff1s)
        cf1, cf1_std = Statistic(cf1s)
        pf1 = pb.TTest_P_Value(ff1s, cf1s)

        flf, flf_std = Statistic(flfs)
        clf, clf_std = Statistic(clfs)
        plf = pb.TTest_P_Value(flfs, clfs)

        fkf, fkf_std = Statistic(fkfs)
        ckf, ckf_std = Statistic(ckfs)
        pkf = pb.TTest_P_Value(fkfs, ckfs)

        lr, lr_std = Statistic([float(rate.split(',')[0]) for rate in rates])
        kr, kr_std = Statistic([float(rate.split(',')[1]) for rate in rates])

        string = ''
        string += '{:5s} | {}    | '.format(configuration_name[:3].upper(), len(results))
        string += '{:06.2%}±{:06.2%} | {:06.2%}±{:06.2%}({:+07.2%})(p={:4.2f}) |    '.format(ff1, ff1_std, cf1, cf1_std, cf1-ff1, pf1)
        string += '{:+06.2f}±{:+06.2f} | {:+06.2f}±{:05.2f}({:+06.2f})(p={:4.2f}) |    '.format(flf, flf_std, clf, clf_std, clf-flf, plf)
        string += '{:+06.2f}±{:+06.2f} | {:+06.2f}±{:05.2f}({:+06.2f})(p={:4.2f}) | '.format(fkf, fkf_std, ckf, ckf_std, ckf-fkf, pkf)
        string += '{:+06.2f}±{:+06.2f} | {:+06.2f}±{:06.2f} | '.format(lr, lr_std, kr, kr_std)
        print(string)

        factual_ans.append([ff1, flf, fkf])
        counterfactual_ans.append([cf1, clf, ckf])
        p_ans.append([pf1, plf, pkf])

        final_map[configuration_name] = [ff1, cf1, flf, clf, fkf, ckf]

    matrix = [final_map[configuration_name] for configuration_name in final_map.keys()]
    if len(matrix)!=0:
        a = np.average(matrix, axis=0)
        print('Total: {} configurations | {:.2%} | {:06.2%}({:+06.2%}) |    {:.4f} | {:.4f}({:+.4f}) |    {:.4f} | {:.4f}({:+.4f})'.format(len(matrix), a[0], a[1], a[1]-a[0], a[2], a[3], a[3]-a[2], a[4], a[5], a[5]-a[4]))
    print()

    return factual_ans, counterfactual_ans, p_ans

def Print_Latex(fs, cs, ps, ispercent=False, smaller=False):
    if len(fs)==0:
        return

    fs.append(np.average(fs))
    cs.append(np.average(cs))
    ps.append(np.average(ps))

    for i,value in enumerate(fs):
        value = value * (100.0 if ispercent==True else 1.0)
        print('{:05.2f}'.format(value), end='')

        func = max if smaller==False else min
        if fs[i] == func(fs[i], cs[i]):
            print('$_*', end='')
            if 'nan' not in str(ps[i]) and ps[i]<=0.05:
                print('^\\dagger', end='')
            print('$', end='')

        if i!=len(fs)-1:
            print(' & ', end='')
        else:
            print(' \\\\')

    for i,value in enumerate(cs):
        value = value * (100.0 if ispercent==True else 1.0)
        print('{:05.2f}'.format(value), end='')

        func = max if smaller==False else min
        if cs[i] == func(fs[i], cs[i]):
            print('$_*', end='')
            if 'nan' not in str(ps[i]) and ps[i]<=0.05:
                print('^\\dagger', end='')
            print('$', end='')

        if i!=len(fs)-1:
            print(' & ', end='')
        else:
            print(' \\\\\n')

# configuration->model
factual_ans, counterfactual_ans, p_ans = Final_Statistic('TextCNN')
Print_Latex([objs[0] for objs in factual_ans], [objs[0] for objs in counterfactual_ans], [objs[0] for objs in p_ans],  ispercent=True)
Print_Latex([objs[1] for objs in factual_ans], [objs[1] for objs in counterfactual_ans], [objs[1] for objs in p_ans],  smaller=True)
Print_Latex([objs[2] for objs in factual_ans], [objs[2] for objs in counterfactual_ans], [objs[2] for objs in p_ans],  smaller=True)

factual_ans, counterfactual_ans, p_ans = Final_Statistic('RoBERTa')
Print_Latex([objs[0] for objs in factual_ans], [objs[0] for objs in counterfactual_ans], [objs[0] for objs in p_ans],  ispercent=True)
Print_Latex([objs[1] for objs in factual_ans], [objs[1] for objs in counterfactual_ans], [objs[1] for objs in p_ans],  smaller=True)
Print_Latex([objs[2] for objs in factual_ans], [objs[2] for objs in counterfactual_ans], [objs[2] for objs in p_ans],  smaller=True)
