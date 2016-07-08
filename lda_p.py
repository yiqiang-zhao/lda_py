#!/usr/bin/python

# use folding-in tech, more details please see: 
# <http://stats.stackexchange.com/questions/9315/topic-prediction-using-latent-dirichlet-allocation>

import copy
import random

voc_items = []
voc_map   = {}

n_kt = []
n_k  = []
K = V = 0
iter_num = 20
alpha = 1
beta  = 0.1

ac_nkt = {}
ac_nk  = {}

def load_dic(fn):
    global voc_items, voc_map
    fi = open(fn, "r")
    for line in fi:
        line = line.strip()
        items = line.split("\t")

        if len(items) != 2:
            print "[ERR Format in Dic]: %s" % line
            continue

        wid = int(items[0].strip())
        w   = items[1].strip()

        voc_items.append(w)
        voc_map[w] = wid

        if wid != len(voc_items)-1:
            print "[ERR: loose some word] (%d,%d) %s" % (wid, len(voc_items)-1, w) 
    fi.close()

def load_stat(fn):
    global K, V, n_k, n_kt

    fi = open(fn, "r")

    i = 0
    for line in fi:
        tmpli = line.strip().split(" ")
        tmp_n_li = map(lambda x:int(x.strip()), tmpli)

        if i == 0:
            if len(tmp_n_li) != 2:
                print "[ERR: bad KV line] %s" % line
            K = tmp_n_li[0]
            V = tmp_n_li[1]
        elif i == 1:
            if len(tmp_n_li) != K:
                print "[ERR: bad n_k line] %d/%d" % (len(tmp_n_li), K)
            n_k = tmp_n_li
        else: # n_kt
            if len(tmp_n_li) != V:
                print "[ERR: n_kt line not match voc] topic %d: %d/%d" % (len(n_kt), len(tmp_n_li), V)
            n_kt.append(tmp_n_li)

        i += 1
    if len(n_kt) != K:
        print "[ERR: n_kt not enough topic lines] %d/%d" % (len(n_kt), K)
    fi.close()

def wl2idl(wl):
    global voc_map
    ret = []
    for w in wl:
        w = w.strip()
        if not voc_map.has_key(w):
            continue
        ret.append(voc_map[w])
    return ret

def inverse_df(pm):
    tmp = pm[:]
    i = 1
    while i < len(pm):
        tmp[i] += tmp[i-1]
        i += 1
    u = random.uniform(0,1) * tmp[len(tmp)-1]
    for index in range(len(tmp)):
        if tmp[index] > u:
            break
    return index

def compute_p(n_mk, t):
    global K, V, n_kt, n_k, alpha, beta
    ret = []
    for k in range(K):
        n1 = float(beta) + n_kt[k][t]
        d1 = float(beta) + n_k[k]
        n2 = float(alpha) + n_mk[k]
        ret.append(n1/d1 * n2)
    return ret

def reg_action(k, t, delta):
    global ac_nkt, ac_nk

    if not ac_nkt.has_key(k):
        ac_nkt[k] = {}
    if not ac_nkt[k].has_key(t):
        ac_nkt[k][t] = 0
    ac_nkt[k][t] += delta

    if not ac_nk.has_key(k):
        ac_nk[k] = 0
    ac_nk[k] += delta

def reset_stat():
    global n_k, n_kt, ac_nkt, ac_nk

    for k in ac_nk:
        n_k[k] -= ac_nk[k]
    for k in ac_nkt:
        for t in ac_nkt[k]:
            n_kt[k][t] -= ac_nkt[k][t]

    ac_nkt.clear()
    ac_nk.clear()

def predict(widl):
    global K, V, n_k, n_kt, iter_num, ac_nkt, ac_nk

    # init
    z    = []
    n_mk = []
    n_m  = 0
    for i in range(K):
        n_mk.append(0)

    l = len(widl)
    for n in range(l):
        t = widl[n]
        k = random.randint(0, K-1)
        z.append(k)

        n_mk[k] += 1; n_m += 1
        n_kt[k][t] += 1; n_k[k] += 1
        reg_action(k, t, 1)

    # samp
    for i in range(iter_num):
        for n in range(l):
            t = widl[n]
            k = z[n]

            n_mk[k] -= 1; n_m -= 1;
            n_kt[k][t] -= 1; n_k[k] -= 1
            reg_action(k, t, -1)

            p = compute_p(n_mk, t)
            k_tilde = inverse_df(p)
            z[n] = k_tilde

            n_mk[k_tilde] += 1; n_m += 1;
            n_kt[k_tilde][t] += 1; n_k[k_tilde] += 1
            reg_action(k_tilde, t, 1)

    reset_stat()
    return z

def tidl2str(tidl):
    ret = ""
    for tid in tidl:
        ret += "%d " % tid
    return ret

def tidl2dist(tidl):
    ret = {}
    cnt = len(tidl)
    if cnt == 0:
        return ret
    for tid in tidl:
        if not ret.has_key(tid):
            ret[tid] = 0.0
        ret[tid] += 1.0
    for tid in ret:
        ret[tid] /= float(cnt)
    return ret

def tid_dis2str(tid_dis):
    ret = ""
    for k,v in sorted(tid_dis.items(), key=lambda x:x[1], reverse=True):
        ret += "%d:%f " % (k, v)
    return ret

def process(fnd, fna, fnt):
    fi = open(fnd, "r")
    fo1 = open(fna, "w")
    fo2 = open(fnt, "w")
    for line in fi:
        li = line.split("\t")
        if len(li) != 2:
            print "[ERR: illegal doc] %s" % line
            continue
        doc_no = int(li[0].strip())
        word_list = li[1].strip().split(" ")
        wid_list = wl2idl(word_list)

        tidl = predict(wid_list)

        tidls = tidl2str(tidl)
        fo1.write("%d\t%s\n" % (doc_no, tidls))

        tid_dis = tidl2dist(tidl)
        tid_dis_str = tid_dis2str(tid_dis)
        fo2.write("%d\t%s\n" % (doc_no, tid_dis_str))

    fi.close()
    fo1.close()
    fo2.close()

def main():
    path = "./db_data/"

    fn_dic  = path + "xaaab_lda_dic.dat"
    fn_stat = path + "xaaab_lda_stat.dat"
    fn_data = path + "xaaaa"
    fn_assign  = path + "xaaaa_p_assign.dat"
    fn_top_dis = path + "xaaaa_p_td.dat"

    load_dic(fn_dic)
    load_stat(fn_stat)
    process(fn_data, fn_assign, fn_top_dis)

if __name__ == "__main__":
    main()
