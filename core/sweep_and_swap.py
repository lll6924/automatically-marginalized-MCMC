from rule import Variable, update, linear, is_linear, is_dependent

def topo_sort(rvs):
    in_degree = {}
    back_ref = {}
    for i in range(len(rvs)):
        in_degree[rvs[i]['name']] = 0
        back_ref[rvs[i]['name']] = i
    for rv in rvs:
        for d in rv['children']:
            in_degree[d] += 1
    queue = []
    index = 0
    for key, val in in_degree.items():
        if val == 0:
            queue.append(key)
    while index < len(queue):
        for d in rvs[back_ref[queue[index]]]['children']:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                queue.append(d)
        index += 1
    ordering = {}
    n = len(queue)
    r_topo_order = []
    for i in range(len(queue)):
        ordering[queue[i]] = i
        r_topo_order.append(queue[n - 1 - i])
    return queue, r_topo_order, ordering

def sweep_and_swap(rvs, name_mapping, expr_mapping, observed, candidates, values, variables, protected):
    topo_order, r_topo_order, ordering = topo_sort(rvs)
    recovery_stack = []
    for r in r_topo_order:

        a = rvs[name_mapping[r]]
        if observed[a['name']] or a['name'] in protected:
            print('Skipping', a['name'])
            continue
        r_dist = rvs[name_mapping[r]]['dist']
        marginalizable = True
        c_sorted = sorted(a['children'], key=lambda x: ordering[x])
        save1 = {}
        save2 = {}
        for c in c_sorted:
            if not c in candidates:
                continue
            c_dist = rvs[name_mapping[c]]['dist']
            b = rvs[name_mapping[c]]
            a_expr = a['expr']
            if r_dist == 'Normal' and c_dist == 'Normal':
                mean, std = variables[b['expr']].parameters
                (x, y, p), save1 = is_linear(mean, variables[a_expr],save1)
                dep, save2 = is_dependent(std, variables[a_expr],save2)
                if not dep and p and values[a['name']].shape == values[b['name']].shape:
                    print('Conjugacy detected between', a['name'], 'and', b['name'])
                else:
                    marginalizable = False
                    print('Conjugacy not detected between', a['name'], 'and', b['name'])
                    break
            elif r_dist == 'Gamma' and c_dist in ['Gamma', 'Exponential']:
                if c_dist == 'Gamma':
                    alpha, beta = variables[b['expr']].parameters
                else:
                    alpha = variables['1.']
                    beta = variables[b['expr']].parameters[0]
                (x, y, p), save1 = is_linear(beta, variables[a_expr],save1)
                dep, save2 = is_dependent(alpha, variables[a_expr],save2)
                if not dep and p and not y and x and values[a['name']].shape == values[b['name']].shape:
                    print('Conjugacy detected between', a['name'], 'and', b['name'])
                else:
                    marginalizable = False
                    print('Conjugacy not detected between', a['name'], 'and', b['name'])
                    break
            elif r_dist == 'Beta' and c_dist in ['BernoulliProbs', 'BinomialProbs']:
                if c_dist == 'Binomial':
                    lamb, cnt = variables[b['expr']].parameters
                else:
                    cnt = variables['1']
                    lamb = variables[b['expr']].parameters[0]
                (x, y, p), save1 = is_linear(lamb, variables[a_expr],save1)
                dep, save2 = is_dependent(cnt, variables[a_expr],save2)
                if not dep and p and not y and x and values[a['name']].shape == values[b['name']].shape:
                    print('Conjugacy detected between', a['name'], 'and', b['name'])
                else:
                    marginalizable = False
                    print('Conjugacy not detected between', a['name'], 'and', b['name'])
                    break
            else:
                marginalizable = False
                break
        if marginalizable:
            save = {}
            for c in c_sorted:
                if not c in candidates:
                    continue

                b = rvs[name_mapping[c]]
                for v in a['dep']:
                    if not v in b['dep']:
                        b['dep'].append(v)
                        rvs[name_mapping[v]]['children'].append(b['name'])
                for v in b['dep']:
                    if not v in a['dep'] and v != a['name']:
                        a['dep'].append(v)
                        rvs[name_mapping[v]]['children'].append(a['name'])
                a['dep'].append(b['name'])
                b['dep'].remove(a['name'])
                b['children'].append(a['name'])
                a['children'].remove(b['name'])
                v1 = a['expr']
                v2 = b['expr']
                c_dist = rvs[name_mapping[c]]['dist']
                if r_dist == 'Normal' and c_dist == 'Normal':
                    mean1, std1 = variables[v1].parameters
                    mean2, std2 = variables[v2].parameters
                    (x, y), save = linear(mean2, variables[v1],save)
                    assert (x is not None and y is not None)
                    z = Variable()  # xm
                    update([z], 'mul', [x, mean1])
                    mean2_new = Variable()  # xm+y
                    update([mean2_new], 'add', [z, y])
                    xx = Variable()  # x^2
                    update([xx], 'square', [x])
                    ss1 = Variable()  # s_1^2
                    update([ss1], 'square', [std1])
                    ss2 = Variable()  # s_2^2
                    update([ss2], 'square', [std2])
                    xs1 = Variable()  # xs_1
                    update([xs1], 'mul', [x, std1])
                    xs1s = Variable()  # x^2s_1^2
                    update([xs1s], 'square', [xs1])
                    xs1ss2 = Variable()  # x^2s_1^2+s_2^2
                    update([xs1ss2], 'add', [xs1s, ss2])
                    std2_new = Variable()
                    update([std2_new], 'sqrt', [xs1ss2])
                    variables[v2].parameters = (mean2_new, std2_new)
                    ss1x = Variable()  # s_1^2x
                    update([ss1x], 'mul', [ss1, x])
                    k = Variable()
                    update([k], 'div', [ss1x, xs1ss2])
                    r = Variable()
                    update([r], 'sub', [variables[v2], mean2_new])
                    kr = Variable()
                    update([kr], 'mul', [k, r])
                    mean1_new = Variable()
                    update([mean1_new], 'add', [mean1, kr])
                    kx = Variable()
                    update([kx], 'mul', [k, x])
                    one_kx = Variable()
                    update([one_kx], 'sub', [Variable('1.0', 1.), kx])
                    one_kx_sqrt = Variable()
                    update([one_kx_sqrt], 'sqrt', [one_kx])

                    std1_new = Variable()
                    update([std1_new], 'mul', [one_kx_sqrt, std1])
                    variables[v1].parameters = (mean1_new, std1_new)

                elif r_dist == 'Gamma' and c_dist in ['Gamma', 'Exponential']:
                    alpha1, beta1 = variables[v1].parameters
                    if c_dist == 'Gamma':
                        alpha2, beta2 = variables[v2].parameters
                    else:
                        beta2 = variables[v2].parameters[0]
                        alpha2 = variables['1.']
                    (x, y), save = linear(beta2, variables[v1],save)
                    assert (x is not None and y == variables['0.'])
                    beta1_transformed = Variable()
                    update([beta1_transformed], 'div', [beta1, x])
                    variables[v2].parameters = (alpha2, alpha1, beta1_transformed)
                    rvs[name_mapping[c]]['dist'] = 'CompoundGamma'
                    alpha1_updated = Variable()
                    update([alpha1_updated], 'add', [alpha1, alpha2])
                    x2_transformed = Variable()
                    update([x2_transformed], 'mul', [variables[v2], x])
                    beta1_updated = Variable()
                    update([beta1_updated], 'add', [beta1, x2_transformed])
                    variables[v1].parameters = (alpha1_updated, beta1_updated)
                elif r_dist == 'Beta' and c_dist in ['BernoulliProbs', 'BinomialProbs']:
                    alpha, beta = variables[v1].parameters
                    if c_dist == 'BinomialProbs':
                        lamb, cnt = variables[v2].parameters
                    else:
                        lamb = variables[v2].parameters[0]
                        cnt = variables['1']
                    (x, y), save = linear(lamb, variables[v1],save)
                    assert (x == variables['1.'] and y == variables['0.'])
                    variables[v2].parameters = (alpha, beta, cnt)
                    rvs[name_mapping[c]]['dist'] = 'BetaBinomial'
                    alpha_updated = Variable()
                    update([alpha_updated], 'add', [alpha, variables[v2]])
                    failed = Variable()
                    update([failed], 'sub', [cnt, variables[v2]])
                    beta_updated = Variable()
                    update([beta_updated], 'add', [beta, failed])
                    variables[v1].parameters = (alpha_updated, beta_updated)
            candidates.remove(a['name'])
            recovery_stack.append(a['name'])
            print(a['name'], "is marginalized")

    return rvs, candidates, variables, recovery_stack