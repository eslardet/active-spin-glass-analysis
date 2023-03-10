import numpy as np
import gzip
import shutil


# with open('coupling', 'rb') as f_in, gzip.open('coupling.gz', 'wb') as f_out:
#     shutil.copyfileobj(f_in, f_out)

# content = "Lots of content here"
# f = gzip.open('zip_test/test2.gz', 'wb')
# f.write((str(3.1415) + '\n').encode())
# f.write(str(2.3442).encode())
# f.close()

f = gzip.open('zip_test/test.gz', 'rb')
file_content = f.read()
print(file_content.decode())