done_parts_folder_path = 'emails/preds_with_0_rag/'
part_i = 1
preds = []
while True:
  try:
    done_part_file_path = done_parts_folder_path + f'preds_with_0_rag_part_{part_i}txt'
    with open(done_part_file_path, 'r') as f:
      lines = f.readlines()
      print(f"{done_part_file_path} - {lines}")
      for line in lines:
        line = line.strip()
        if line != '':
          preds.append(int(line))
  except Exception as e:
    print(f"No Part - {part_i}")
    break
  part_i += 1
print(len(preds))