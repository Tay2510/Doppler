//
//  ViewController.m
//  Doppler 
//
//  modified by Chao-Ming Yen from  Kurt Jacobs's code
//

#import "ViewController.h"

#define KBYTES_CLEAN_UP 10000 //10 Megabytes Max Storage Otherwise Force Cleanup (For This Example We Will Probably Never Reach It -- But Good Practice).
#define LUAT_STACK_INDEX_FLOAT_TENSORS 4 //Index of Garbage Collection Stack Value

@interface ViewController ()
{
    CvPhotoCamera *photoCamera_; // OpenCV wrapper class to simplfy camera access through AVFoundation
    BOOL isLive;
}
@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
  
    // main view
    //*******************************************************
    self.resultView.hidden = true;
    [self.view addSubview:self.mainView]; // Important: add liveView_ as a subview
    [self.view addSubview:self.resultView];
    [self.camButton setTitle:@"Take picture" forState:UIControlStateNormal];
    
    // 4. Initialize the camera parameters and start the camera (inside the App)
    photoCamera_ = [[CvPhotoCamera alloc] initWithParentView:self.mainView];
    photoCamera_.delegate = self;
    
    // This chooses whether we use the front or rear facing camera
    photoCamera_.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    
    // This is used to set the image resolution
    photoCamera_.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    
    // This is used to determine the device orientation
    photoCamera_.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    
    // This starts the camera capture
    [photoCamera_ start];
    isLive = true;
    
    //*******************************************************
    
    // creating Torch virtual machine
    //*******************************************************
    self.t = [Torch new];
    [self.t initialize];
    [self.t runMain:@"main" inFolder:@"torch7"];
    [self.t loadFileWithName:@"vgg_normalized.th" inResourceFolder:@"torch7" andLoadMethodName:@"loadModel"];
    //*******************************************************
}

// To be compliant with the CvPhotoCameraDelegate we need to implement these two methods
- (void)photoCamera:(CvPhotoCamera *)photoCamera capturedImage:(UIImage *)image
{
    [photoCamera_ stop];
    self.mainView.hidden = true;
    self.resultView.hidden = false;
    
    cv::Mat cvImage; UIImageToMat(image, cvImage);
    cv::transpose(cvImage, cvImage); // correct rotation
    UIImage *resImage = MatToUIImage(cvImage);
    
    // Special part to ensure the image is rotated properly when the image is converted back
    self.resultView.image =  [UIImage imageWithCGImage:[resImage CGImage]
                                                 scale:1.0
                                           orientation:UIImageOrientationUp];
}

- (IBAction)onClicked:(id)sender {
    if (isLive) {
        [photoCamera_ takePicture];
        [self.camButton setTitle:@"Live" forState:UIControlStateNormal];
        isLive = false;
    } else {
        [photoCamera_ start];
        self.mainView.hidden = false;
        self.resultView.hidden = true;
        [self.camButton setTitle:@"Take picture" forState:UIControlStateNormal];
        isLive = true;
    }
}

- (void)photoCameraCancel:(CvPhotoCamera *)photoCamera
{
    
}

// Torch part
// ******************************************************************************************
// ******************************************************************************************
- (BOOL)isValidFloat:(NSString*)string
{
  NSScanner *scanner = [NSScanner scannerWithString:string];
  [scanner scanFloat:NULL];
  return [scanner isAtEnd];
}

- (void)perfClassificationOnValuesv1:(float)v1 v2:(float)v2
{
  XORClassifyObject *classificationObj = [XORClassifyObject new];
  classificationObj.x = v1;
  classificationObj.y = v2;
  float value = [self classifyExample:classificationObj inState:[self.t getLuaState]];
  //self.answerLabel.text = [NSString stringWithFormat:@"Classification Value: %f",value];
}

- (CGFloat)classifyExample:(XORClassifyObject *)obj inState:(lua_State *)L
{
  NSInteger garbage_size_kbytes = lua_gc(L, LUA_GCCOUNT, LUAT_STACK_INDEX_FLOAT_TENSORS);

  if (garbage_size_kbytes >= KBYTES_CLEAN_UP)
  {
    NSLog(@"LUA -> Cleaning Up Garbage");
    lua_gc(L, LUA_GCCOLLECT, LUAT_STACK_INDEX_FLOAT_TENSORS);
  }

  THFloatStorage *classification_storage = THFloatStorage_newWithSize1(2);
  THFloatTensor *classification = THFloatTensor_newWithStorage1d(classification_storage, 1, 2, 1);
  THTensor_fastSet1d(classification, 0, obj.x);
  THTensor_fastSet1d(classification, 1, obj.y);
  lua_getglobal(L,"classifyExample");
  luaT_pushudata(L, classification, "torch.FloatTensor");
  
  //p_call -- args, results
  int res = lua_pcall(L, 1, 1, 0);
  if (res != 0)
  {
    NSLog(@"error running function `f': %s",lua_tostring(L, -1));
  }
  
  if (!lua_isnumber(L, -1))
  {
    NSLog(@"function `f' must return a number");
  }
  CGFloat returnValue = lua_tonumber(L, -1);
  lua_pop(L, 1);  /* pop returned value */
  return returnValue;
}
// ******************************************************************************************
// ******************************************************************************************

- (void)didReceiveMemoryWarning
{
  [super didReceiveMemoryWarning];
}

@end
