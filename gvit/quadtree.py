import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt

class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def set_area(self, mask, patch):
        patch_size = self.get_size()
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2
    
    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1
    
    def draw(self, ax, c='grey', lw=0.5, **kwargs):
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
    
    
class QuadTree:
    def __init__(self, domain, bbox = None, depth=0, max_value=80, max_depth=6) -> None:
        # default to NxN size
        self.domain = domain
        self.bbox = bbox
        self.depth = depth
        self.max_value = max_value
        self.max_depth = max_depth
        self.divided = False
        self.childs = None
        
    
        self._build_tree()
        
    def get_rect(self):
        return self.bbox.get_coord()
    
    def serialize(self, img):
        seq_patch = []
        if self.divided:
            for c_n in self.childs:
                seq_patch += c_n.serialize(img)
        else:
            seq_patch.append(self.bbox.get_area(img))
        return seq_patch
    
    def deserialize(self, seq, mask):
        if self.divided:
            for c in self.childs:
                total_patches +=  c.deserialize(seq, mask)
        else:
            pred_mask = seq.pop(0)
            self.bbox.set_area(mask, pred_mask)
        return mask
    
    def count_patches(self, patch_info):
        total_patches = 0
        if self.divided:
            for c in self.childs:
                total_patches +=  c.count_patches(patch_info)
        else:
            patch_size = self.bbox.get_size()
            key = "{}*{}".format(patch_size[0],patch_size[1])
            if key not in patch_info.keys():  
                patch_info[key] = 1
            else:
                patch_info[key] += 1
            return 1
        return total_patches
    
    def _build_tree(self):
        if self.bbox == None:
            h,w = self.domain.shape
            assert h>0 and w >0, "Wrong img size."
            self.bbox = Rect(0,w,0,h)
        value = self.bbox.contains(self.domain)
        if value > self.max_value and self.depth<self.max_depth:
            self.divide()
            
    def divide(self):
        x1,x2,y1,y2 = self.bbox.get_coord()
        
        # bbox of the current node.
        self.lt = QuadTree(self.domain, Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2), self.depth + 1, self.max_value, self.max_depth)
        self.rt = QuadTree(self.domain, Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2), self.depth + 1, self.max_value, self.max_depth)
        self.lb = QuadTree(self.domain, Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2)), self.depth + 1, self.max_value, self.max_depth)
        self.rb = QuadTree(self.domain, Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2)), self.depth + 1, self.max_value, self.max_depth)
                                  
        self.divided = True
        self.childs = [self.lt, self.rt, self.lb, self.rb]
        
    def draw(self, ax, c='grey', lw=1, **kwargs):
        self.bbox.draw(ax=ax)
        if self.divided:
            for c_n in self.childs:
                c_n.draw(ax=ax)
                
class FixedQuadTree:
    def __init__(self, domain, fixed_length=128) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()
        
    def _build_tree(self):
    
        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        self.nodes = [(root, root.contains(self.domain))]
        while len(self.nodes)<self.fixed_length:
            bbox, value = self.nodes.pop()
        
            x1,x2,y1,y2 = bbox.get_coord()
            lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
            v1 = lt.contains(self.domain)
            rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
            v2 = rt.contains(self.domain)
            lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
            v3 = lb.contains(self.domain)
            rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
            v4 = rb.contains(self.domain)
            
            self.nodes += [(lt,v1), (rt,v2), (lb,v3), (rb,v4)]
            self.nodes = sorted(self.nodes, key=lambda x: x[1])
            # print([v for _,v in self.nodes])
            
    def count_patches(self):
        return len(self.nodes)
    
    def serialize(self, img, size=(8,8,3)):
        
        seq_patch = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            
        h2,w2,c2 = size
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            assert h1==w1, "Need squared input."
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
            assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
            
        return seq_patch
    
    def deserialize(self, seq, mask):
        for bbox,value in self.nodes:
            pred_mask = seq.pop(0)
            mask = bbox.set_area(mask, pred_mask)
        return mask
    
    def draw(self, ax, c='grey', lw=1):
        for bbox,value in self.nodes:
            bbox.draw(ax=ax)
    
                
