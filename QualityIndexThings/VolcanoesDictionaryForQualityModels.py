#This set of dictionaries hold information specific to each camera
#location/viewing-angle/time-period combination e.g. the registration
#transform used, path to matching clear-sky images etc. Only the
#information relevant to the example application scripts is
#included here.

#This work contains fine-tuned precipitation and cloud filtering neural network models,
#intended for quality classification of UV SO2 Camera video data.
#Copyright (C) 2026 Alyssa Heggison

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see https://www.gnu.org/licenses/.

#Please contact: asheggison1@sheffield.ac.uk, or through GitHub: https://github.com/AHeggison/QualityIndexModelsodels


Cotopaxi_View2_dictionary = {'volcano_dictionary_name':"CotopaxiView2",
                             'volcano_name':"Cotopaxi",
                             'sensor_marks_mask_A':"None",
                             'sensor_marks_mask_B':"None"}

Cotopaxi_View3_dictionary = {'volcano_dictionary_name':"CotopaxiView3",
                             'volcano_name':"Cotopaxi",
                             'sensor_marks_mask_A':"None",
                             'sensor_marks_mask_B':"None"}

Cotopaxi_View4_dictionary = {'volcano_dictionary_name':"CotopaxiView4",
                             'volcano_name':"Cotopaxi",
                             'sensor_marks_mask_A': "None",
                             'sensor_marks_mask_B': "None"}

Kilauea_View1_dictionary = {'volcano_dictionary_name':"KilaueaView1",
                            'volcano_name':'Kilauea',
                            'sensor_marks_mask_A': "Kilauea_1A.png",
                            'sensor_marks_mask_B': "Kilauea_1B.png"}

Kilauea_View2_dictionary = {'volcano_dictionary_name':"KilaueaView2",
                            'volcano_name':'Kilauea',
                            'sensor_marks_mask_A': "None",
                            'sensor_marks_mask_B': "None"}

Reventador_2022_dictionary = {'volcano_dictionary_name':"Reventador2022",
                              'volcano_name':'Reventador',
                              'sensor_marks_mask_A': "Reventador_2022A.png",
                              'sensor_marks_mask_B': "None"}


Reventador_2023_dictionary = {'volcano_dictionary_name':"Reventador2023",
                              'volcano_name':'Reventador',
                              'sensor_marks_mask_A': "Reventador_2024A.png",
                              'sensor_marks_mask_B': "None"}

Reventador_2024_dictionary = {'volcano_dictionary_name':"Reventador2024",
                              'volcano_name':'Reventador',
                              'sensor_marks_mask_A': "Reventador_2024A.png",
                              'sensor_marks_mask_B': "None"}

Lastarria_dictionary = {'volcano_dictionary_name':"Lastarria",
                        'volcano_name':'Lastarria',
                        'sensor_marks_mask_A': "Lastarria_A.png",
                        'sensor_marks_mask_B': "Lastarria_B.png"}


Merapi_View0_dictionary = {'volcano_dictionary_name':"MerapiView0",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_TenthMay_dictionary = {'volcano_dictionary_name':"MerapiTenthMay",
                              'volcano_name':'Merapi',
                              'sensor_marks_mask_A': "Merapi_5A.png",
                              'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_SixteenthMay_dictionary = {'volcano_dictionary_name':"MerapiSixteenthMay",
                                  'volcano_name':'Merapi',
                                  'sensor_marks_mask_A': "Merapi_5A.png",
                                  'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_TwSecondJune_dictionary = {'volcano_dictionary_name':"MerapiTwSecondJune",
                                  'volcano_name':'Merapi',
                                  'sensor_marks_mask_A': "Merapi_5A.png",
                                  'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_SecondOct_dictionary = {'volcano_dictionary_name':"MerapiSecondOct",
                               'volcano_name':'Merapi',
                               'sensor_marks_mask_A': "Merapi_5A.png",
                               'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_View1_dictionary = {'volcano_dictionary_name':"MerapiView1",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_View2_dictionary = {'volcano_dictionary_name':"MerapiView2",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_View3_dictionary = {'volcano_dictionary_name':"MerapiView3",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_3B.png"}

Merapi_View4_dictionary = {'volcano_dictionary_name':"MerapiView4",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_4B.png"}

Merapi_View5_dictionary = {'volcano_dictionary_name':"MerapiView5",
                           'volcano_name':'Merapi',
                           'sensor_marks_mask_A': "Merapi_5A.png",
                           'sensor_marks_mask_B': "Merapi_5B.png"}

all_dictionaries = [Cotopaxi_View2_dictionary, Cotopaxi_View3_dictionary, Cotopaxi_View4_dictionary, Kilauea_View1_dictionary, Kilauea_View2_dictionary, Reventador_2022_dictionary,
                    Reventador_2023_dictionary, Reventador_2024_dictionary, Lastarria_dictionary, Merapi_View0_dictionary, Merapi_View1_dictionary, Merapi_View2_dictionary,
                    Merapi_TenthMay_dictionary, Merapi_SixteenthMay_dictionary, Merapi_TwSecondJune_dictionary, Merapi_SecondOct_dictionary, Merapi_View3_dictionary,
                    Merapi_View4_dictionary, Merapi_View5_dictionary]

all_dictionary_names = []
for dictionary in all_dictionaries:
    all_dictionary_names.append(dictionary['volcano_dictionary_name'])

def map_dictionary_name_to_dictionary(dictionary_name):
    index = all_dictionary_names.index(dictionary_name)
    return all_dictionaries[index]