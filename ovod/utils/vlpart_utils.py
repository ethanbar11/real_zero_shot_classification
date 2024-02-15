from types import SimpleNamespace
import os

path_to_grounding_ge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DEFAULT_VLPART_ARGS = SimpleNamespace(
    config_file=os.path.join(path_to_grounding_ge_dir,
                             'third_party/VLPart/configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml'),
    webcam=False,
    video_input=None,
    input=[],
    output='output_image',
    vocabulary='custom',
    custom_vocabulary='flower',
    confidence_threshold=0.6,
    opts=[
        'MODEL.WEIGHTS',
        '/shared-data5/SHARED-MODELS/VLPart/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth'
        # os.path.join(path_to_grounding_ge_dir,
                     # 'third_party/VLPart/models/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth')
    ]
)
