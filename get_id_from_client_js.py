import sys

print('get_id_from_client_js.py running ... ')
msg = sys.argv[1]
if msg is not None:
    file_ = open(r'../id_from_client_js.txt', 'w')
    file_.write(msg)
    print(f'text file saved')
    file_.close()
