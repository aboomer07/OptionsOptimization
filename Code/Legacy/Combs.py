def get_combos(size_lim):

	opt_dict = {'sc' : {'left' : 0, 'right' : -1},
	    'lc' : {'left' : 0, 'right' : 1},
	    'sp' : {'left' : -1, 'right' : 0},
	    'lp' : {'left' : 1, 'right' : 0}}

	def valid_opts(opt_dict, combo, size, curr_size):
	    bound = {'left' : 0, 'right' : 0}
	   
	    for pos in combo:
	        for key, val in opt_dict[pos].items():
	            bound[key] += val
	   
	    valid = ['sc', 'lc', 'sp', 'lp']
	   
	    rem = size - curr_size
	   
	    if (bound['left'] + rem <= 1):
	        valid = [i for i in valid if i != 'sp']
	    if (bound['right'] + rem <= 1):
	        valid = [i for i in valid if i != 'sc']
	    if bound['right'] + rem == 0:
	        valid = [i for i in valid if i != 'lp']
	        valid = [i for i in valid if i != 'sp']
	    if bound['left'] + rem == 0:
	        valid = [i for i in valid if i != 'lc']
	        valid = [i for i in valid if i != 'sc']
	   
	    return(valid)

	combos = {}

	for size in range(1, (size_lim + 1)):
	    curr_size = 0
	    valid = valid_opts(opt_dict, [], size, curr_size)
	    combos['Size' + str(size)] = [[i] for i in valid]
	    curr_size += 1
	   
	    while curr_size < size:
	        valids = []
	       
	        for combo in combos['Size' + str(size)]:
	            valid = valid_opts(opt_dict, combo, size, curr_size)
	            valids.append(valid)
	       
	        new_combos = []
	        for i in range(len(valids)):
	            old = combos['Size' + str(size)][i]
	            new = valids[i]
	            new_combos.extend([old + [val] for val in new])
	       
	        combos['Size' + str(size)] = new_combos
	        curr_size += 1
	
	return(combos)




	