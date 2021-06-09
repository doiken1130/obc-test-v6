# os 非依存的WEBアプリケーション
import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
import time
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import itertools #
import openpyxl

st.title("アイくん -乳がん超音波検査サポート-")
# st.title("乳がん超音波検査サポート")
st.sidebar.title("ツールメニュー")

# mode=st.sidebar.selectbox(" ツールの選択 ", ("技師名を編集する", "AIによる画像診断", "診断情報の登録", "データベース照会", "相性解析"))
mode=st.sidebar.selectbox(" ツールの選択 ", ("技師名を編集する", "AIによる画像診断", "診断情報の登録", "データベース照会", "相性解析", "初期化（開発者のみ使用可）"))

classes=['negative', 'positive']
size=256

# root_dir=os.getcwd()
root_dir="/Users/doikentaro/Desktop/obc_web_v6"
posi_dir=root_dir+"//image_store//positive//"
nega_dir=root_dir+"//image_store//negative//"
tmp_dir=root_dir+"//tmp//" # root_dir//tmp//
tmp_posi_dir=tmp_dir+"positive//" # root_dir//tmp//positive//
tmp_nega_dir=tmp_dir+"negative//" # root_dir//tmp//negative//
tec_name_dir=root_dir+"//tec_names.csv"
result_dir=root_dir+"//result_examine.xlsx"
prediction_dir=root_dir+"//prediction_result.xlsx"
json_path = root_dir+"//result_examine.json" #
model_path=root_dir+"//neural_network//outperform_model//vgg16_ft.h5"

path_list=[posi_dir, nega_dir, tmp_dir, tmp_posi_dir, tmp_nega_dir]
storing_path=[tmp_posi_dir, tmp_nega_dir]

def download_tec_names():
    tec_name=pd.read_csv(tec_name_dir, header=None, skiprows=1, dtype=str, encoding='utf-8')
    tec_name=tec_name[1]
    return tec_name


if mode == "診断情報の登録":
    st.title(" 超音波検査情報の入力 ")
    col1, col2 = st.beta_columns(2)
    information=[]


    with col1:

        st.write('<span><b><u>診断情報の登録</b></u></span>', unsafe_allow_html=True)

        if os.name == 'nt':
            tmp_pd = pd.read_excel(prediction_dir)#windows
            ttt_pd = pd.read_excel(result_dir)#windows

        else:
            tmp_pd = pd.read_excel(prediction_dir, engine='openpyxl') #mac
            ttt_pd = pd.read_excel(result_dir, engine='openpyxl') #mac

        patient_select_pd = tmp_pd['患者ID']
        patient_id = st.selectbox("該当する患者IDを選択してください", patient_select_pd)
        # if ttt_pd['患者ID'].str.__contains__(patient_select_pd) is True: #要修正箇所
        #     st.write("既に登録されている患者IDが重複しています。IDを選択しなおさない場合、ID-1 の様にIDを保存します。")


        information.append(patient_id)

        monngonn=st.selectbox("文言を選択", ("LNmeta疑い", "LNmetaあり", "良悪判定つかず", "反応性LN疑い"))

        information.append(monngonn)

        syuyoukei=st.text_input("原発腫瘍径を入力" )


        information.append(syuyoukei)

        nyuugann_cate=st.selectbox("乳腺カテゴリを選択", (" 1 ", " 2 "," 3 "," 4 "," 5 "))
        information.append(nyuugann_cate)


        T_hanntei=st.selectbox("T判定を選択", (" T1 ", " T2以上 ", "-"))
        information.append(T_hanntei)

        tyoukei=st.text_input("長径を入力" )

        information.append(tyoukei)

        tannkei=st.text_input("短径を入力")

        information.append(tannkei)

        LS=st.text_input("L/Sを入力")

        information.append(LS)

        LS_2=st.selectbox("L/S < 2", (" ○ ", " X "))

        information.append(LS_2)

        new_LNkeijou=st.selectbox(" 新LN形状を選択 ", (" I ", " II "," III "," IV "," V "," VI "))
        information.append(new_LNkeijou)

        LN_mon=st.selectbox("LN門", ("消失", "あり"))
        information.append(LN_mon)

    with col2:

        ketsuryuu=st.selectbox("血流部位を選択", ("(+:皮質)", "(+:皮質以外)", "(-),計測なし"))
        information.append(ketsuryuu)

        tahatsu=st.selectbox(" 多発か選択 ", ("-", "多発"))
        information.append(tahatsu)

        heikei = st.selectbox("閉経の状況について選択", ("閉経前", "閉経後", "閉経なし"))
        information.append(heikei)

        age = st.text_input(" 患者の年齢を入力 ")
        information.append(age)

        FNA = st.selectbox("FNA", ("実施meta(+)", "実施meta(-)", "実施なし"))
        information.append(FNA)

        chemo=st.selectbox("ケモ", ("(+)", "(-)"))
        information.append(chemo)

        byouri = st.selectbox("病理", ("n1以上", "n0"))
        information.append(byouri)

        meta = st.selectbox("meta (FNAor病理)", ("有", "無"))
        information.append(meta)

    st.write('<span><b><u>技師の登録</b></u></span>', unsafe_allow_html=True)


    tec_name=download_tec_names() #リストで帰ってくる
    a=st.selectbox("放射線技師の選択", tec_name)
    information.append(a)

    mes_1="<span><u><b>"+ a +"</b></u></span>"

    st.write('<span style=color:red><b>技師の編集は、左のツールバーの<u>”技師の編集”</u>から行ってください</b></span>', unsafe_allow_html=True)



    if st.button(" データベースに登録する ") is True:
        result_dir = root_dir + "//result_examine.xlsx"

        if os.name == 'nt':
            read_pd=pd.read_excel(result_dir) # dataframeDOYONA...
        else:
            read_pd=pd.read_excel(result_dir,engine='openpyxl') # dataframeDOYONA...


        json_pd = pd.DataFrame({
            "患者ID": ['test'],
            "文言": ['test'],
            "原発腫瘍径": ['test'],
            "乳腺カテゴリー": ['test'],
            "T判定": ['test'],
            "長径": ['test'],
            "短径": ['test'],
            "L/S": ['test'],
            "L/S < 2": ['test'],
            "新LN形状": ['test'],
            "LN門": ['test'],
            "血流部位": ['test'],
            "多発": ['test'],
            "閉経": ['test'],
            "年齢": ['test'],
            "FNA": ['test'],
            "ケモ": ['test'],
            "病理": ['test'],
            "meta（FNA or 病理）": ['test'],
            "放射線技師": ['test']
        }, dtype=str)

        for ind, koumoku in enumerate(json_pd):
            json_pd[koumoku]=json_pd[koumoku].replace('test', information[ind])
        result_pd=read_pd.append(json_pd)
        result_pd.to_excel(result_dir, index=False)

        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'データベース登録 {i + 1}% 完了')
            bar.progress(i + 1)
            time.sleep(0.01)



if mode == "AIによる画像診断":
    st.title("アイくんによる画像診断")
    im_name=st.file_uploader(" 参照ボタンまたはドラッグアンドドロップにより画像をアップロードできます ")

    if im_name is not None:
        data_size=(256,256)
        img_zero = Image.open(im_name)
        img=img_zero.convert('RGB')
        img_array=np.array(img)
        data=cv2.resize(img_array, data_size, interpolation=cv2.INTER_CUBIC)



        information_2=[]


        if os.name == 'nt':
            patient_info_diagnosed=pd.read_excel(prediction_dir)
        else:
            patient_info_diagnosed=pd.read_excel(prediction_dir,engine='openpyxl')

        patient_info_id_toroku=st.text_input("登録する患者IDを入力してください")
        patient_info_diagnosed_id=patient_info_diagnosed['患者ID']


        image_name=st.text_input("画像の保存名を入力してください")
        image_name=image_name+".jpg"
        if st.button("画像を保存する") is True:
            for path in storing_path:
                store_name=path+image_name
                cv2.imwrite(store_name, data)  # debug
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.text(f'画像保存 {i + 1}% 完了')
                bar.progress(i + 1)
                time.sleep(0.01)

        st.image(data, caption="対象の画像", use_column_width=True)

        tec_name = download_tec_names()  # リストで帰ってくる
        a_ = st.selectbox("放射線技師の選択", tec_name)


        if st.button(" AIで画像診断をする ") is True:
            tmp_dir=tmp_dir
            test_generator=ImageDataGenerator(rescale=1. /255)
            test_generated=test_generator.flow_from_directory(
                tmp_dir,
                target_size=(size, size),
                batch_size=1,
                color_mode='rgb',
                classes=classes,
                class_mode='binary',
                shuffle=False
            )

            AI_model = load_model(model_path)
            tmp_pred=AI_model.predict(test_generated)
            pred = []
            data_name = test_generated.filenames
            inference_list = []

            if tmp_pred[0] > 0.5:
                present = "悪性"
                percentage = tmp_pred * 100
                percentage=percentage[0]

                inference_list.append(1)
                pred.append(tmp_pred)
            else:
                present = "良性"
                percentage = (1 - tmp_pred) * 100
                percentage=percentage[0]

                inference_list.append(0)
                pred.append(tmp_pred)
            percentage = str(percentage).replace("[", "")
            percentage = str(percentage).replace("]", "")

            message=(present+"である可能性が高いです。"+percentage+"% の確率で、"+present+"であると判断しました。")
            st.write(message)

            data_zero = np.array(img_zero)
            data_zero = cv2.resize(data_zero, data_size, interpolation=cv2.INTER_CUBIC)  # 保存前の画像処理

            if tmp_pred[0] > 0.5:
                im_save_name = posi_dir + '//' + image_name

                if os.path.exists(im_save_name):
                    dirpath, filename = os.path.split(im_save_name)
                    name, ext = os.path.splitext(filename)

                    for i in itertools.count(1):
                        newname = '{} -case{} {}'.format(name, i, ext)
                        new_im_save_name = os.path.join(dirpath, newname)
                        if not os.path.exists(new_im_save_name):
                            break
                    cv2.imwrite(new_im_save_name, data_zero)
                else:
                    cv2.imwrite(im_save_name, data_zero)  # ok
                message_2 = ("Positive (悪性)のフォルダに画像を保存しました。")
            else:
                im_save_name = nega_dir + "//" + image_name
                if os.path.exists(im_save_name):
                    dirpath, filename = os.path.split(im_save_name)
                    name, ext = os.path.splitext(filename)

                    for i in itertools.count(1):
                        newname = '{} -case{} {}'.format(name, i, ext)
                        new_im_save_name = os.path.join(dirpath, newname)
                        if not os.path.exists(new_im_save_name):
                            break
                    cv2.imwrite(new_im_save_name, data_zero)
                else:
                    cv2.imwrite(im_save_name, data_zero)  # ok
                message_2 = ("Negative (良性)のフォルダに画像を保存しました。")
            st.write(message_2)

            information_2.append(patient_info_id_toroku)

            inf_pred=str(tmp_pred[0])
            inf_pred=inf_pred.replace("[", "")
            inf_pred=inf_pred.replace("]", "")
            information_2.append(inf_pred)

            information_2.append(present)

            information_2.append(image_name)
            information_2.append(a_)
            print(information_2)

            if os.name == 'nt':
                read_ai_inference=pd.read_excel(prediction_dir)
            else:
                read_ai_inference=pd.read_excel(prediction_dir, engine='openpyxl')

            ai_inference = pd.DataFrame({
                    "患者ID": ['test'],
                    "推定値": ['test'],
                    "推定結果": ['test'],
                    "画像名": ['test'],
                    "放射線技師": ['test']
                }, dtype=str)
            print(ai_inference)
            for ind, koumoku in enumerate(ai_inference):
                ai_inference[koumoku] = ai_inference[koumoku].replace('test', information_2[ind])

            read_ai_inference=read_ai_inference.append(ai_inference)
            read_ai_inference.to_excel(prediction_dir, index=False)

            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.text(f'診断結果保存 {i + 1}% 完了')
                bar.progress(i + 1)
                time.sleep(0.01)

if mode == "技師名を編集する":
    st.title(" 技師名を編集する ")

    st.write('<span style=color:red><b>**登録技師と削除技師は同時に入力しないでください**</b></span>', unsafe_allow_html=True)

    if st.button(" 技師名簿を参照する ") is True:
        st.write("現在、登録されている技師名は以下の通りです")
        tec_name = download_tec_names()

        tec_name_list = list(tec_name)

        l = len(tec_name_list)
        st.write(pd.DataFrame(tec_name))

    new_technitian_name = st.text_input(" 追加する技師の名前を入力 --例-- 土井")
    tec_name = download_tec_names()

    tec_name_list=list(tec_name)

    l=len(tec_name_list)
    if tec_name_list.__contains__(new_technitian_name):
        st.write(" すでに登録されている技師名であるため、登録できません ")
    else:

        tec_name_list.append(new_technitian_name)

        df_tec_names=pd.DataFrame(tec_name_list)

        if st.button(" 登録 ") is True:
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
                latest_iteration.text(f'技師名登録 {i + 1}% 完了')
                bar.progress(i + 1)
                time.sleep(0.01)
            df_tec_names.to_csv(tec_name_dir, encoding='utf-8')

    removed_technitian_name = st.text_input(" 削除する技師の名前を入力 --例-- 土井")
    tec_name = download_tec_names()

    tec_name_list=list(tec_name)

    l=len(tec_name_list)
    if st.button(" 削除 "):
        try:
            tec_name_list.remove(removed_technitian_name)
            df_tec_names = pd.DataFrame(tec_name_list)
            df_tec_names = df_tec_names.dropna()

            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
                latest_iteration.text(f'技師名削除 {i + 1}% 完了')
                bar.progress(i + 1)
                time.sleep(0.01)
            df_tec_names.to_csv(tec_name_dir, encoding='utf-8')

        except ValueError:
            st.write(" 入力された技師は本システムに未登録であるため、削除することができません ")
            st.write(" もう一度入力しなおしてください ")

if mode == "AIをアップデートする":
    st.title("AIをアップデートする")

if mode == "データの解析":
    st.title("データを解析する")
    st.write(" 機械学習や統計学的処理によりデータを解析します ")

if mode == "開発者情報をみる":
    st.title(" 開発者情報 ")
    st.write("開発者：土井健太郎")
    st.write("所属：都島放射線科クリニック研究生 / 大阪大学大学院医学系研究科博士課程")
    st.write("*本アプリの二次配布はお断りしています")
    st.write("問い合わせ先：kentaro.doi@sahs.med.osaka-u.ac.jp")

if mode == "データベース照会":
    if os.name == 'nt':
        tmp_pd_syoukai = pd.read_excel(result_dir)  # windows
    else:
        tmp_pd_syoukai = pd.read_excel(result_dir, engine='openpyxl')  # mac

    patient_select_pd_syoukai = tmp_pd_syoukai['患者ID']
    patient_id_syoukai = st.selectbox("該当する患者IDを選択してください", patient_select_pd_syoukai)
    patient_id_syoukai_str=str(patient_id_syoukai)
    for i, id in enumerate(patient_select_pd_syoukai):
        if str(id) == patient_id_syoukai_str:
            index_syoukai = i
    identified_pd=tmp_pd_syoukai.loc[index_syoukai]

    if st.button("照会する") is True:
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'データベース照会 {i + 1}% 完了')
            bar.progress(i + 1)
            time.sleep(0.01)
        st.write("照会結果")
        st.write(identified_pd)

    st.write("※＜診断情報の登録＞にて情報が登録されていない患者IDは検索することができません。")

if mode == "初期化":
    if st.button("初期化する") is True:
        pass
