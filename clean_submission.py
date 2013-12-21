__author__ = 'alex'


if __name__ == '__main__':
    f_out = open('clean_subm.csv','w+')
    f_out.write('"Id","Tags"\n')
    f = 0
    with open('test_result.csv') as f_subm:
        next(f_subm)
        i = 0

        for line in f_subm:

            if '\r' not in line or '\n' not in line:
                splits = []
                try:
                    splits = line.split(',')
                except:
                    continue

                if not splits[0].isdigit():
                    continue

                try:
                    if int(splits[0]) < 6034190 or int(splits[0]) > 8047532:
                        continue
                except:
                    continue

                if splits[1] == '""root\n':
                    line = splits[0] + ',' + '"root"\n'
                    f += 1
                elif len(splits)>2 and int(splits[0]) > 6034196:
                    print line, i
                    line = splits[0] + ',' + '"java"\n'
                    f += 1
                    print "fix ->>", line
                elif '\n' not in line:
                    print line ,i
                    line = splits[0] + ',' + '"java"\n'
                    f += 1
                    print "fix------>" , line

            i +=1
            f_out.write(line)

    f_out.close()
    print f

