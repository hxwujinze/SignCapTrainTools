import json
import os
import pickle
import shutil

import process_data_dev


def main():
    source_dir = 'test_dir'
    source_dir_abs_path = os.path.join(process_data_dev.DATA_DIR_PATH, source_dir)
    stat_book = process_data_dev.statistics_data(source_dir)
    target_dir_abs_path = os.path.join(process_data_dev.DATA_DIR_PATH, 'cleaned_data')
    if not os.path.exists(target_dir_abs_path):
        os.makedirs(target_dir_abs_path)

    scanned_book = {}
    scanned_book_f = 'scanned_data.dat'
    if os.path.exists(scanned_book_f):
        f = open(scanned_book_f, 'r+b')
        scanned_book = pickle.load(f)

    print(json.dumps(scanned_book, indent=2))

    for each_sign in stat_book.keys():
        for each_batch in stat_book[each_sign]['occ_pos']:
            try:
                scanned_book[each_sign].index(each_batch)
                continue
            except (ValueError, KeyError):
                pass

            each_batch = each_batch.split(' ')
            date = each_batch[0]
            batch_id = each_batch[1]
            print("show data sign %s %s %s" % (each_sign, date, batch_id))
            process_data_dev.print_train_data(sign_id=int(each_sign),
                                              batch_num=int(batch_id),
                                              data_cap_type='acc',
                                              data_feat_type='poly_fit',
                                              capture_date=date,
                                              data_path=source_dir,
                                              for_cnn=True)

            process_data_dev.print_train_data(sign_id=int(each_sign),
                                              batch_num=int(batch_id),
                                              data_cap_type='gyr',
                                              data_feat_type='poly_fit',
                                              capture_date=date,
                                              data_path=source_dir,
                                              for_cnn=True)
            mark = '%s %s' % (date, batch_id)
            if scanned_book.get(each_sign) is None:
                scanned_book[each_sign] = [mark]
            else:
                scanned_book[each_sign].append(mark)

            print("save it? y/n")
            res = input()
            if res == 'y':
                for each_type in ['Acceleration', 'Emg', 'Gyroscope']:
                    source_file_path = os.path.join(date, str(batch_id), each_type)

                    old_path = os.path.join(source_dir_abs_path, source_file_path, str(each_sign) + '.txt')
                    target_path = os.path.join(target_dir_abs_path, source_file_path)

                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    new_path = os.path.join(target_path, str(each_sign) + '.txt')
                    shutil.copyfile(old_path, new_path)

            with open(scanned_book_f, 'w+b') as f:
                pickle.dump(scanned_book, f)


if __name__ == '__main__':
    main()
