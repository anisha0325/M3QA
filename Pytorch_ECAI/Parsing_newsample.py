###################### Test file parsing #############################

filepath = "MedQA_dataset/newsample.txt"
with open(filepath, 'r') as file:
    content = file.readlines()


contexts = []
for elem in content:
  if 'Context' in elem:
    contexts.append(elem.strip())
print(len(contexts))

def split_on_context(lst):
    result = []
    current_context = None

    for item in lst:
        if item.startswith('[Context '):
            # If we find a new context, start a new sublist
            if current_context is not None:
                result.append(current_context)
            current_context = [item.strip()]
        else:
            # Otherwise, append the item to the current context list
            if current_context is not None and item != '\n':
                current_context.append(item.strip())

    # Don't forget to append the last context list to the result
    if current_context is not None:
        result.append(current_context)

    return result

# Example usage:
result = split_on_context(content)
print(len(result))

def list_to_dict(lst):
    current_id = None
    current_entry = {}
    result = {}
    for item in lst:
        key, value = item.split(' --> ')
        key = key.strip()
        value = value.strip()

        if key == 'QID':
            # When encountering a new ID, store the current entry if it exists
            if current_id is not None:
                result[current_id] = current_entry
            # Start a new entry
            current_id = value
            current_entry = {}
        else:
            # For 'que' and 'ans', add to the current entry
            current_entry[key] = value

    # Don't forget to add the last entry to the result
    if current_id is not None:
        result[current_id] = current_entry

    return result

context_qa_list = []
for res in result:
  context_dict = {res[0]: res[1]}
  qa_dict =  list_to_dict(res[2:])
  context_qa_list.append({**context_dict, **qa_dict})

import pickle

file_path = 'context_qa_list.pkl'

# Dump the list to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(context_qa_list, file)


from tqdm import tqdm
QID_context_q = dict()
QID_ans = dict()
QID_ques = dict()
QID_context = dict()

for elem in tqdm(context_qa_list):
  for key, value in elem.items():
    if 'Context' in key:
      present_context = value
    else:
      QID = key
      ques = value['Que']
      ans = value['Ans']
      QID_context_q[QID] = ques + ' [SEP] ' + present_context + ' [CLS] '
      QID_ans[QID] = ans
      QID_ques = ques
      QID_context[QID] = present_context

with open('QID_q_context.pkl', 'wb') as file:
    pickle.dump(QID_context_q, file)

with open('QID_ans.pkl', 'wb') as file:
    pickle.dump(QID_ans, file)

with open('QID_ques.pkl', 'wb') as file:
    pickle.dump(QID_ques, file)

with open('QID_context.pkl', 'wb') as file:
    pickle.dump(QID_context, file)

