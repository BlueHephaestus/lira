"""
Calls our main function in microscopic_prediction_editor with all our default values.

If you wish to modify these default values, then modify them.

-Blake Edwards / Dark Element
"""

import microscopic_prediction_editor

microscopic_prediction_editor.edit_microscopic_predictions(sub_h=80, 
         sub_w=145, 
         img_archive_dir="../lira/lira1/data/images.h5",
         resized_img_archive_dir="../lira/lira1/data/resized_images.h5",
         predictions_archive_dir="../lira/lira1/data/predictions.h5",
         classification_metadata_dir="../lira_static/classification_metadata.pkl",
         interactive_session_metadata_dir="interactive_session_metadata.pkl",
         live_archive_dir="../lira/lira2/data/live_samples.h5",
         dual_monitor=False,
         resize_factor=0.1,
         save_new_samples=False,
         rgb=True)

