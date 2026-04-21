# Standard class mappings for the 4 hyperspectral datasets
DATASET_CLASS_NAMES = {
        'indian_pines': {
            0: 'Background', 1: 'Alfalfa', 2: 'Corn-notill', 3: 'Corn-mintill', 
            4: 'Corn', 5: 'Grass-pasture', 6: 'Grass-trees', 7: 'Grass-pasture-mowed',
            8: 'Hay-windrowed', 9: 'Oats', 10: 'Soybean-notill', 11: 'Soybean-mintill',
            12: 'Soybean-clean', 13: 'Wheat', 14: 'Woods', 15: 'Buildings-Grass-Trees-Drives',
            16: 'Stone-Steel-Towers'
        },
        'salinas_valley': {
            0: 'Background', 1: 'Brocoli_green_weeds_1', 2: 'Brocoli_green_weeds_2', 
            3: 'Fallow', 4: 'Fallow_rough_plow', 5: 'Fallow_smooth', 6: 'Stubble', 
            7: 'Celery', 8: 'Grapes_untrained', 9: 'Soil_vinyard_develop', 
            10: 'Corn_senesced_green_weeds', 11: 'Lettuce_romaine_4wk', 
            12: 'Lettuce_romaine_5wk', 13: 'Lettuce_romaine_6wk', 14: 'Lettuce_romaine_7wk', 
            15: 'Vinyard_untrained', 16: 'Vinyard_vertical_trellis'
        },
        'pavia_university': {
            0: 'Background', 1: 'Asphalt', 2: 'Meadows', 3: 'Gravel', 4: 'Trees',
            5: 'Painted metal sheets', 6: 'Bare Soil', 7: 'Bitumen',
            8: 'Self-Blocking Bricks', 9: 'Shadows'
        },
        'pavia_center': {
            0: 'Background', 1: 'Water', 2: 'Trees', 3: 'Asphalt', 
            4: 'Self-Blocking Bricks', 5: 'Bitumen', 6: 'Tiles', 7: 'Shadows', 
            8: 'Meadows', 9: 'Bare Soil'
        }
    }


# dataset paths
DATASET_PATHS = {
        'indian_pines': {
            'data': 'ds/indian_pines/indian_pines_corrected.mat',
            'gt': 'ds/indian_pines/indian_pines_gt.mat'
        },
        'salinas_valley': {
            'data': 'ds/salinas_valley/salinas_valley_corrected.mat',
            'gt': 'ds/salinas_valley/salinas_valley_gt.mat'
        },
        'pavia_center': {
            'data': 'ds/pavia_center/pavia_center.mat',
            'gt': 'ds/pavia_center/pavia_center_gt.mat'
        },
        'pavia_university': {
            'data': 'ds/pavia_university/pavia_university.mat',
            'gt': 'ds/pavia_university/pavia_university_gt.mat'
        }
    }