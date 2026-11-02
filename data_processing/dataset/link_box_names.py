# coding:utf-8
"""Rename box labels that name a title entity in a different written form.

The data pipeline links a title mention to a box only when the box label equals
the mention string. Some boxes name the same person or place as a mention but
in another form: a role name without its epithet, an actor name with a
generation prefix, or a variant character. This script writes a copy of the
annotation directory in which such labels carry the title string, so that the
mention counts as depicted. Boxes that name something the title does not
mention (for example Mount Fuji drawn in a print whose title does not name it)
keep their label and stay unlinked.

Usage: python3 data_processing/dataset/link_box_names.py [src_dir] [dst_dir]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_processing import config as cfg     # every path of the project
import os, sys, shutil, copy
import xml.etree.ElementTree as ET


# (print, box label) -> title strings it names; reviewed by hand, one by one
LINK = {
    ('arcUP0007', '与次郎兵衛'): ['笛吹与次郎兵衛'],
    ('arcUP4678', '重兵衛'): ['ごふくや重兵衛'],
    ('NDL-459-02-097', '政尾'): ['中老政尾'],
    ('NDL-143-07-085', '五郎市'): ['一子五郎市'],
    ('shiUYa0094', '伊左衛門'): ['藤屋伊左衛門'],
    ('BM-paul0084', '五代目松本幸四郎'): ['松本幸四郎'],
    ('NDL-540-00-016', '門跡'): ['築地門跡'],
    ('NDL-143-10-020', '小町屋宗七'): ['宗七'],
    ('arcUP2189', '宗七'): ['小町屋宗七'],
    ('MM0609_62', '筑紫の権六'): ['権六'],
    ('NDL-115-01-011', '惣太'): ['猿島惣太'],
    ('NDL-426-00-010', '妙椿'): ['尼妙椿'],
    ('NDL-143-04-020', '時次郎'): ['春日屋時次郎'],
    ('NDL-143-04-020', '浦里'): ['山名屋浦里'],
    ('Z0165-387', '長五郎'): ['ぬれかみ長五郎'],
    ('Z0165-387', '大谷広次'): ['大谷広治'],
    ('arcUP0336', '甘輝'): ['五常軍甘輝'],
    ('Z0170-008', '富士'): ['富嶽'],
    ('Z0170-137', '富士'): ['富嶽'],
    ('Z0170-130', '富士'): ['富嶽'],
    ('Z0172-032', '富士'): ['冨士'],
    ('arcUY0235', '晴光'): ['小森半之丞晴光'],
    ('arcUY0235', '小森半之丞'): ['小森半之丞晴光'],
    ('arcUP0057', '因幡ノ助正忠'): ['正忠'],
    ('NDL-159-00-026', '黒主'): ['偽ス黒主'],
    ('NDL-143-02-054', '金子'): ['いやみ金子'],
    ('NDL-143-02-054', '金之助'): ['いやみ金之助'],
    ('arcUP6615', '小万'): ['だつきの小万'],
    ('MM0637_65', '姉宮城野'): ['宮城野'],
    ('arcUY0342', '市村羽左衛門'): ['羽左衛門'],
    ('arcUY0342', '中村芝翫'): ['芝翫'],
    ('NDL-296-00-005', '万代橋'): ['筋違万代橋'],
    ('AkoRH-RG001-54', '音吉'): ['鳶の者音吉'],
    ('NDL-460-01-047', '惣太'): ['さるしま惣太'],
    ('Z0172-697', '福岡みつぎ'): ['みつぎ'],
    ('Ebi0740(02)', '福岡みつぎ'): ['みつぎ'],
    ('NDL-152-00-008', '菅神'): ['菅神化身'],
    ('nakai014-2', '小町桜のせい'): ['せい'],
    ('arcUP4506', '下りさだ吉'): ['さだ吉'],
    ('arcUP4506', 'げい者おくに'): ['おくに'],
    ('arcUY0105', '雷鶴之助'): ['角力取雷鶴之助'],
    ('arcUP3968', '駒太郎'): ['手綱駒太郎'],
    ('NDL-158-00-036', '権兵衛'): ['うつらの権兵衛'],
    ('NDL-143-10-050', '長九郎'): ['手代長九郎'],
    ('arcUP5152', '松や手代忠兵衛'): ['忠兵衛'],
    ('NDL-105-02-051', 'おくま'): ['御守殿おくま'],
    ('NDL-112-01-068', '米吉'): ['百金の米吉'],
    ('MMS040', '津の国長次'): ['長次'],
    ('BA49607901-2-05', '市川団十郎'): ['市川團十郎'],
    ('NDL-108-01-032', '繁之丞'): ['三蔭山繁之丞'],
    ('NDL-152-00-026', '高貞'): ['塩冶判官高貞'],
    ('NDL-152-00-026', '待つ豊'): ['山名家全待つ豊'],
    ('NDL-152-00-026', '与次兵衛'): ['山崎屋与次兵衛'],
    ('NDL-295-00-001', '眼鏡橋'): ['筋違眼鏡橋'],
    ('arcBK01-0041_22', '伝八'): ['番頭伝八'],
    ('NDL-116-00-014', '矢藤与茂七'): ['与茂七'],
    ('arcUY0180', 'よし兵へ'): ['梅のよし兵へ'],
    ('arcUP6581', '忍売おくみ'): ['おくみ'],
    ('arcBK01-0044_74', '進'): ['蔵ノ進'],
    ('nakai020-1', '新左衛門'): ['鬼王新左衛門'],
    ('Z0172-676', '長吉'): ['放駒長吉'],
    ('NDL-108-02-004', '青砥孫三郎源藤太常景」'): ['青砥孫三郎', '源藤太常景'],
    ('NDL-224-02-004', '五郎兵衛'): ['八百屋五郎兵衛'],
    ('Z0165-354', '鎌田次郎'): ['鎌田次郎道清'],
    ('Z0165-354', '道清'): ['鎌田次郎道清'],
    ('MMS313_01', '団十郎'): ['団十郎改'],
    ('NDL-460-02-058', 'おりつ'): ['ぎおんのおりつ'],
    ('NDL-402-00-004', '小町'): ['七小町'],
    ('Nishiy-002', '団十郎'): ['団十郎七'],
    ('NDL-108-03-012', 'おりつ'): ['ぎおんのおりつ'],
    ('arcBK01-0044_17', '喜兵へ'): ['はんじ物喜兵へ'],
    ('arcUP2406', '伝吉'): ['七面伝吉'],
    ('NDL-124-01-041', '庵椿やま'): ['椿やま'],
    ('NDL-460-02-061', 'おたの'): ['ごぜおたの'],
    ('NDL-112-01-040', 'おきく'): ['相つちのおきく'],
    ('NDL-151-00-011', '伊左衛門'): ['藤屋伊左衛門'],
    ('arcUP4272', '尾上多賀之丞'): ['多賀之丞'],
    ('NDL-143-03-014', 'お六'): ['土手のお六'],
    ('Ebi0664', '鶴之介'): ['三ヶつ地鶴之介'],
    ('NDL-134-01-009', 'さゞゐ堂'): ['五百羅漢さゞゐ堂」'],
    ('arcUP4509', '長五郎'): ['四ッイ菱長五郎'],
    ('NDL-108-03-032', '駄右衛門'): ['日本駄右衛門'],
    ('NDL-115-02-050', 'おしげか'): ['おしげ'],
    ('arcUP0332', '梅'): ['寺島の梅'],
    ('arcUP0332', '竜'): ['音羽の竜'],
    ('NDL-113-00-053', '初代尾上菊次郎'): ['尾上菊次郎'],
    ('NDL-528-00-005', 'つぎ橋'): ['真間つぎ橋'],
    ('NDL-108-06-019', '頼兼'): ['左金吾頼兼'],
    ('NDL-113-00-007', '十代目羽左衛門'): ['羽左衛門'],
    ('NDL-113-00-007', '八代目羽左衛門'): ['羽左衛門'],
    ('NDL-113-00-007', '四代目市村竹之丞'): ['市村竹之丞'],
    ('MMS137_01', '政子御前'): ['政子'],
    ('NDL-112-07-016', '由利の八郎'): ['八郎'],
    ('arcBK01-0043-11', '悪七兵衛'): ['悪七兵衛景清'],
    ('arcBK01-0043-11', '日向景清'): ['悪七兵衛景清'],
    ('NDL-186-00-009', '四天王東門'): ['四天王'],
    ('shiUY0263', '漢ノ張良'): ['張良'],
    ('NDL-138-00-004', '男山'): ['枚方男山'],
    ('NDL-152-00-028', '三代目八百蔵'): ['八百蔵'],
    ('NDL-152-00-028', '二代目八百蔵'): ['八百蔵'],
    ('NDL-152-00-028', '初代市川八百蔵'): ['市川八百蔵'],
    ('NDL-143-06-037', 'いかみの権太'): ['権太'],
    ('NDL-143-02-040', 'お辰'): ['一寸嶋お辰'],
    ('Z0169-279', '藤蔵'): ['奴藤蔵'],
    ('NDL-446-00-003', '数奇屋'): ['数寄屋'],
    ('arcUP2808', '浜名や半七'): ['半七'],
    ('arcUP0187', '光然'): ['仏光寺光然'],
    ('NDL-181-00-004', '良雄'): ['大星由良之助良雄'],
    ('NDL-181-00-004', '大星由良之助'): ['大星由良之助良雄'],
    ('shiUY0228', '在原業平'): ['内業平'],
    ('shiUY0228', '小町'): ['内小町'],
    ('Ebi1361', '乕蔵'): ['艸履取乕蔵'],
    ('NDL-112-05-003', '中老尾のゑ'): ['尾のゑ'],
    ('NDL-460-02-070', 'となせ'): ['女房となせ'],
}


def main(src=cfg.XML_DIR, dst=cfg.XML_LINKED_DIR):
    if not os.path.isdir(src):     # check before dst is removed
        sys.exit('source directory not found: %s' % src)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    used = set()
    for (img, label), targets in LINK.items():
        path = os.path.join(dst, img + '.xml')
        tree = ET.parse(path)
        root = tree.getroot()
        for obj in list(root.findall('object')):
            name = obj.find('name')
            if name is None or name.text is None or name.text.strip() != label:
                continue
            used.add((img, label))
            name.text = targets[0]
            for extra in targets[1:]:   # one box that names two title entities
                dup = copy.deepcopy(obj)
                dup.find('name').text = extra
                root.append(dup)
        tree.write(path, encoding='utf-8', xml_declaration=True)
    missing = set(LINK) - used
    assert not missing, missing
    print('relabelled %d box labels in %s' % (len(used), dst))


if __name__ == '__main__':
    main(*sys.argv[1:])
