from pyrouge import Rouge155

r = Rouge155()
custom_rouge_args = '-e /Users/catherinebaker/evaluation/ROUGE-RELEASE-1.5.5/data -n 2 -a -l 100'
r.system_dir = '/Users/catherinebaker/Desktop/571labs/HW3/System_Summaries'
r.model_dir = '/Users/catherinebaker/Desktop/571labs/HW3/Human_Summaries/eval'
r.system_filename_pattern = 'd3(#ID#)t.'
r.model_filename_pattern = 'D3(#ID#).M.100.T.(\w)'

#systems = ['Centroid', 'DPP', 'ICSISumm', 'LexRank', 'Submodular']
system = 'Centroid' # changed this variable per system ROUGE check
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

r.system_dir = '/Users/catherinebaker/Desktop/571labs/HW3/System_Summaries' + '/' + system
r.system_filename_pattern = 'd3(\d+)t.' + system
output = r.convert_and_evaluate(rouge_args=custom_rouge_args)
print("\n\n\n" + system + "\n")
print(output)
output_dict = r.output_to_dict(output)