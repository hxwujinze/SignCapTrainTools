import os

import process_data_dev


def main():
    source_dir = 'collect_data_test'
    source_dir_abs_path = os.path.join(process_data_dev.DATA_DIR_PATH, source_dir)
    batch_list = os.listdir(source_dir_abs_path)
    batch_list = [int(each) for each in batch_list]
    last_batch = sorted(batch_list)[-1]

    sign_list = os.listdir(os.path.join(source_dir_abs_path, str(last_batch), 'Emg'))
    sign_list = [int(each.strip('.txt')) for each in sign_list]

    while True:
        if len(sign_list) == 0:
            break

        print("capture signs: %s" % sign_list)
        print('choose which sign to show as num')
        res = input()
        try:
            res = int(res)
        except:
            break

        try:
            sign_list.index(res)
            sign_list.remove(res)
        except ValueError:
            print("no such sign have been collected")
            continue

        process_data_dev.print_train_data(sign_id=res,
                                          batch_num=int(last_batch),
                                          data_cap_type='acc',
                                          data_feat_type='poly_fit',
                                          capture_date=None,
                                          data_path=source_dir,
                                          for_cnn=True)

        process_data_dev.print_train_data(sign_id=int(res),
                                          batch_num=int(last_batch),
                                          data_cap_type='gyr',
                                          data_feat_type='poly_fit',
                                          capture_date=None,
                                          data_path=source_dir,
                                          for_cnn=True)


if __name__ == '__main__':
    main()
