//
//  ViewController.h
//  Doppler
//
//  Created by Tay on 2016/12/6.
//  Copyright © 2016年 CMU. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Torch.h"
#include <Torch/Torch.h>
#import "XORClassifyObject.h"

@interface ViewController : UIViewController

@property (nonatomic, strong) Torch *t;
@property (weak, nonatomic) IBOutlet UILabel *answerLabel;
@property (weak, nonatomic) IBOutlet UITextField *valueOneTextfield;
@property (weak, nonatomic) IBOutlet UITextField *valueTwoTextField;

@end

