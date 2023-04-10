import torch


class TrainGlobalConfigShunt_with_all:
    num_workers = 8
    n_epochs = 16
    lr = 0.002
    
    folder = 'weights/Shunt_with_all/'
    verbose = True
    verbose_step = 1
    
    step_scheduler = False
    validation_scheduler = True
    
    target = 'Shunt'
    
    features = ['BGA_Ca', 'BGA_FiO2_BGA',
       'BGA_Glu', 'BGA_Hb_BGA', 'BGA_HCO3', 'BGA_K', 'BGA_Lac', 'BGA_Na',
       'BGA_PaO2/FiO2', 'BGA_PCO2', 'BGA_pH', 'BGA_PO2', 'BGA_SBE',
       'BGA_sO2', 'Labor_Albumin', 'Labor_Alk._Phospatase',
       'Labor_ALT_(GPT)', 'Labor_aPTT', 'Labor_AST_(GOT)',
       'Labor_Bilirubin_gesamt', 'Labor_CK', 'Labor_CRP',
       'Labor_Erythrocyten', 'Labor_EVB', 'Labor_Fibrinogen', 'Labor_fT3',
       'Labor_fT4', 'Labor_GFR', 'Labor_GGT', 'Labor_Harnstoff',
       'Labor_Hb', 'Labor_Hk', 'Labor_INR', 'Labor_Kreatinin',
       'Labor_LDH', 'Labor_Leukocyten', 'Labor_Lipase', 'Labor_Magnesium',
       'Labor_MCH', 'Labor_MCHC', 'Labor_MCV',
       'Labor_pankreasspez._Amylase', 'Labor_Phosphat_anorg.',
       'Labor_Quick', 'Labor_Thrombocyten', 'Labor_Triglyceride',
       'Labor_TSH_basal', 'Labor_TZ', 'Vital_AF', 'Vital_AMV_total',
       'Vital_ASB', 'Vital_Ausfuhr', 'Vital_Ausfuhr_Magensonde',
       'Vital_AZV', 'Vital_BDK', 'Vital_Beatmungsmodus', 'Vital_Bilanz',
       'Vital_Bilanz_mit_Blut+Liquor', 'Vital_Bilanz_seit_Aufnahme',
       'Vital_Bilanz_um_6:00_Uhr', 'Vital_BIS', 'Vital_diast',
       'Vital_Einfuhr', 'Vital_EVD', 'Vital_EVD_drain',
       'Vital_EVD_niveau', 'Vital_FiO2', 'Vital_Flow_Trigger',
       'Vital_Freq_gesamt', 'Vital_Freq_spontan', 'Vital_GCS_auge',
       'Vital_GCS_motor', 'Vital_GCS_total', 'Vital_GCS_verbal',
       'Vital_HF', 'Vital_I_E', 'Vital_Insp_Druck', 'Vital_Lagerung',
       'Vital_Lichtreaktion_li/re', 'Vital_mittl', 'Vital_oraler_Tubus',
       'Vital_PEEP', 'Vital_Pmean', 'Vital_Ppeak', 'Vital_Pulsation',
       'Vital_Pupille_li', 'Vital_Pupille_re',
       'Vital_Residualvolumen_MS_Ausfuhr',
       'Vital_Residualvolumen_MS_Einfuhr', 'Vital_SpO2', 'Vital_syst',
       'Vital_t_insp', 'Vital_Temp', 'Vital_Urinproduktion', 'Vital_ZVD',
       'BGA_Cl', 'Vital_CPP', 'Vital_ICP', 'Vital_Liquor_Ausfuhr_gesamt',
       'Labor_Cholesterin', 'Labor_PCT', 'Vital_Liquor_Drainage',
       'Vital_Mobilisation', 'Labor_Vancomycin_vor_Gabe',
       'Vital_diast_NBD', 'Vital_mittl_NBD', 'Vital_Resp',
       'Vital_Spontanurin', 'Vital_syst_NBD', 'Vital_ATC',
       'Vital_Tagesbilanz_Ziel', 'Labor_Basophile', 'Labor_Lymphocyten',
       'BGA_Bili', 'BGA_FCOHb', 'Labor_Eiweiß_gesamt',
       'Vital_Kcal_Zufuhr_in_24h', 'Vital_RASS', 'Vital_arterielle_BGA',
       'Vital_Exsp_CO2', 'Vital_Behavioral_Pain_Scale',
       'Labor_CK_MB_(U/l)', 'Labor_Troponin_T', 'Labor_Lactat',
       'Labor_Liquor:_Gesamteiweiß', 'Labor_Liquor:_Glucose',
       'Labor_Liquor:_Granulozyten', 'Labor_Liquor:_IL6',
       'Labor_Liquor:_Lymphozyten', 'Labor_Liquor:_Monozyten',
       'Labor_Liquor:_Zellzahl/Leukozyten', 'Labor_NSE', 'Labor_TSH',
       'Labor_Harnstoff_N', 'Vital_Delir_Score',
       'Labor_Ethanol_enzym._Blut', 'Vital_Dynamische_Compliance',
       'BGA_Temp_BGA', 'Vital_DRG_Einfuhr_Infusionen',
       'Vital_DRG_Ausfuhr_Urin',
       'Vital_DRG_Einfuhr_Perfusor_Kurzinfusion', 'Vital_RASS_Ziel',
       'Vital_Numeric_Rating_Scale', 'Labor_Pat._Temperatur',
       'Vital_Puls', 'Größe', 'Gewicht', 'Alter', 'Geschlecht',
                'HuntandHess','Fisher', 'Graeb'               
               ]

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    
    scheduler_params = dict(
        mode='min',
        factor=0.7,
        patience=5,
        verbose=False,
        threshold=0.01,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0.001,
        eps=1e-08
    )
