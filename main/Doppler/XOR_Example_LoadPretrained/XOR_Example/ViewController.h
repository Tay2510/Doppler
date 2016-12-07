//
//  ViewController.h
//  Doppler
//
//  modified by Chao-Ming Yen from  Kurt Jacobs's code
//

#import <UIKit/UIKit.h>
#import "Torch.h"
#include <Torch/Torch.h>
#import "XORClassifyObject.h"

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/highgui/ios.h"
#endif

@interface ViewController : UIViewController<CvPhotoCameraDelegate>

@property (nonatomic, strong) Torch *t;
@property (weak, nonatomic) IBOutlet UIImageView *mainView;
@property (weak, nonatomic) IBOutlet UIImageView *resultView;
@property (weak, nonatomic) IBOutlet UIButton *camButton;

@end

