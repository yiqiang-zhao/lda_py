#!/usr/bin/python

# Heinrich G. Parameter estimation for text analysis, version 2.9. 
# <http://arbylon.net/publications/text-est2.pdf>, 2009.

import random
import sys

doc_set = []
M = K = V = 0

## count variables
n_mk = [] # document-topic count
n_m  = [] # document-topic sum
n_kt = [] # topic-term count
n_k  = [] # topic-term sum

z = []

num_it = 0 # num of iteration

max_doc_len = 0
alpha = 1
beta = 0.1

# vocabulary
voc_itms = []
voc_map = {}

def get_wids(words):
    global voc_itms, voc_map
    res = []
    for word in words:
        word = word.strip()
        if len(word) == 0:
            continue
        if not voc_map.has_key(word): # insert into voc
            voc_map[word] = len(voc_itms)
            voc_itms.append(word)
        res.append(voc_map[word]) # insert into res
    return res

def save_dic(fn_dic):
    global voc_itms
    fo = open(fn_dic, "w")
    for i in range(len(voc_itms)):
        fo.write("%d\t%s\n" % (i, voc_itms[i]))
    fo.close()

def save_doc(fn_doc):
    global doc_set
    fo = open(fn_doc, "w")
    for i in range(len(doc_set)):
        fo.write("%d\t" % i)
        for wid in doc_set[i]:
            fo.write("%d " % wid)
        fo.write("\n")
    fo.close()

def read_docs(fnd):
    global doc_set, M, V, max_doc_len
    fi = open(fnd, "r")
    print "[Reading in docs]"
    for line in fi:
        li = line.split("\t")
        if len(li) != 2:
            print "illegal doc: %s" % line
            continue

        doc_no = int(li[0].strip())
        words = li[1].strip().split(" ")
        word_ids = get_wids(words)

        if len(word_ids) == 0:
            continue

        doc_set.append(word_ids)

        M += 1
        if len(word_ids) > max_doc_len:
            max_doc_len = len(word_ids)
    fi.close()
    V = len(voc_itms)

def zero_count():
    global n_mk, n_m, n_kt, n_k, M, K, V
    for i in range(M):
        n_m.append(0)
        n_mk.append([])
    for i in range(K):
        n_k.append(0)
        n_kt.append([])
    for i in range(M):
        for j in range(K):
            n_mk[i].append(0)
    for i in range(K):
        for j in range(V):
            n_kt[i].append(0)

def init():
    global doc_set, M, K, z, n_mk, n_m, n_kt, n_k
    # zero all count variable
    zero_count()

    for m in range(M):
	z.append([])
        for n in range(len(doc_set[m])):
            t = doc_set[m][n]

            k = random.randint(0,K-1)
            z[m].append(k)

            n_mk[m][k] += 1
            n_m[m]     += 1
            n_kt[k][t] += 1
            n_k[k]     += 1

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

def compute_p(m, t):
    global K, V, n_mk, n_m, n_kt, n_k, alpha, beta
    ret = []
    for k in range(K):
        n1 = float(beta) + n_kt[k][t]
        d1 = float(beta) + n_k[k]
        n2 = float(alpha) + n_mk[m][k]
        ret.append(n1/d1 * n2)
    return ret

def gibbs_samp():
    global num_it, doc_set, z, M, n_mk, n_m, n_kt, n_k
    for i in range(num_it):
        for m in range(M):
            for n in range(len(doc_set[m])):
                t = doc_set[m][n]
                k = z[m][n]

                n_mk[m][k] -= 1
                n_m[m]     -= 1
                n_kt[k][t] -= 1
                n_k[k]     -= 1

                p = compute_p(m, t)
                k_tilde = inverse_df(p)
                z[m][n] = k_tilde

                n_mk[m][k_tilde] += 1
                n_m[m]           += 1
                n_kt[k_tilde][t] += 1
                n_k[k_tilde]     += 1
        #if is_converge():
        #    break

def export_phi(fn):
    global n_kt, n_k, beta, K, V
    fo = open(fn, "w")
    for k in range(K):
        fo.write("%d " % k)
        for t in range(V):
            n = n_kt[k][t] + beta
            d = n_k[k] + beta
            fo.write("%f " % (float(n)/d))
        fo.write("\n")
    fo.close()

def export_theta(fn):
    global n_mk, n_m, M, K, alpha
    fo = open(fn, "w")
    for m in range(M):
        fo.write("%d " % m)
        for k in range(K):
            n = n_mk[m][k] + alpha
            d = n_m[m] + alpha
            fo.write("%f " % (float(n)/d))
        fo.write("\n")
    fo.close()

def export_z(fn):
    global z, M
    fo = open(fn, "w")
    for m in range(M):
        fo.write("%d " % m)
        for topic_id in z[m]:
            fo.write("%d " % topic_id)
        fo.write("\n")
    fo.close()

def save_stat(fn):
    global K, V, n_kt, n_k
    fo = open(fn, "w")

    fo.write("%d %d\n" % (K, V))

    for k in range(K):
       fo.write("%d " % n_k[k])
    fo.write("\n")

    for k in range(K):
        for t in range(V):
            fo.write("%d " % n_kt[k][t])
        fo.write("\n")

    fo.close()

def main():
    global K, num_it
    path = "./data/"
    fn_doc_set = path + "raws.dat"

    fn_dic = path + "dic.dat"
    fn_doc = path + "docs.dat"

    fn_theta = path + "theta.dat"
    fn_phi = path + "phi.dat"
    fn_tassign = path + "tassign.dat"

    fn_stat = path + "stat.dat"

    K = 50
    num_it = 20

    read_docs(fn_doc_set)
    save_dic(fn_dic)
    save_doc(fn_doc)

    init()
    gibbs_samp()

    export_z(fn_tassign)
    export_phi(fn_phi)
    export_theta(fn_theta)

    save_stat(fn_stat) # for predict

if __name__ == "__main__":
    main()
