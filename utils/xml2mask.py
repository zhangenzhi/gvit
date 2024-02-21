import numpy as np
import xml.etree.ElementTree as et
from scipy.ndimage import binary_fill_holes
fill = binary_fill_holes
from scipy.ndimage import label
from threading import Thread
import os, glob, tifffile, openslide
from threading import Thread, Lock

'''
Annotations (root)
> Annotation (get 'Id' -> 1: whole tumor // 2: viable tumor)
 > Regions
  > Region (get 'NegativeROA' -> 0: positive area // 1: inner negative area)
   > Vertices
    > Vertex (get 'X', 'Y')
'''
pad = 1024
offset = pad

def load_svs_shape(fn, level=0):
    #print('loading shape of <{}> / level={}..'.format(fn, level))
    imgh = openslide.OpenSlide(fn)
    shape = [imgh.level_dimensions[level][1], imgh.level_dimensions[level][0]]
    imgh.close()  # Close the OpenSlide image to release resources
    return shape

def xml2contour(fn, shape, div=None):
  print('reconstructing sparse xml to dense contours..')
  board1 = None
  board2 = None
  board_neg = np.zeros(shape[:2], dtype=np.uint8)
  #y_min = np.inf
  #x_min = np.inf
  #y_max = 0
  #x_max = 0
  # Annotations >> 
  e = et.parse(fn).getroot()
  e = e.findall('Annotation')
  # assert(len(e) == 2), len(e)
  for ann in e:
    board = np.zeros(shape[:2], dtype=np.uint8)
    id_num = int(ann.get('Id'))
    assert(id_num == 1 or id_num == 2)
    regions = ann.findall('Regions')
    assert(len(regions) == 1)
    rs = regions[0].findall('Region')
    for i, r in enumerate(rs):
      ylist = []
      xlist = []
      dys = []
      dxs = []
      negative_flag = int(r.get('NegativeROA'))
      assert(negative_flag == 0 or negative_flag == 1)
      negative_flag = bool(negative_flag)
      vs = r.findall('Vertices')[0]
      vs = vs.findall('Vertex')
      vs.append(vs[0]) # last dot should be linked to the first dot
      for v in vs:
        y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
        y, x = y+offset, x+offset
        if div is not None:
          y //= div
          x //= div
        if y >= shape[0]:
          y = shape[0]-1
        elif y < 0:
          y = 0
        if x >= shape[1]:
          x = shape[1]-1
        elif x < 0:
          x = 0
        if not negative_flag and board[y, x] < 1:
          #board[y, x] = 2 if negative_flag else 1
          board[y, x] = 1
        elif negative_flag and board_neg[y, x] < 1:
          board_neg[y, x] = 1
        #if orderlist and order != 0: # must exclude copied first dot
        #  assert(orderlist[-1] < order), 'orderlist[-1]: {}, order: {}'.format(orderlist[-1], order)
        #orderlist.append(order)
        if len(ylist) > 1 and ylist[-1] == y and xlist[-1] == x:
          continue
        ylist.append(y)
        xlist.append(x)
        #if not negative_flag:
        #  if y < y_min:
        #    y_min = y
        #  elif y > y_max:
        #    y_max = y
        #  if x < x_min:
        #    x_min = x
        #  elif x > x_max:
        #    x_max = x
        if len(ylist) <= 1:
          continue
        #print('y - prev:', y, ylist[-1], 'x - prev:', x, xlist[-1])
        y_prev = ylist[-2]
        x_prev = xlist[-2]
        dy = y - y_prev
        dx = x - x_prev
        y_factor = 1 if dy > 0 else -1
        x_factor = 1 if dx > 0 else -1
        y_cur = y_prev + y_factor
        x_cur = x_prev + x_factor
        it = 0

        if dx == 0:
          if y_factor > 0:
            while y_cur <= y:
              if not negative_flag and board[y_cur, x_cur] < 1:
                #board[y_cur, x_cur] = 2 if negative_flag else 1
                board[y_cur, x_cur] = 1
              elif negative_flag and board_neg[y_cur, x_cur] < 1:
                board_neg[y_cur, x_cur] = 1
              y_cur += y_factor
          else:
            while y_cur >= y:
              if not negative_flag and board[y_cur, x_cur] < 1:
                #board[y_cur, x_cur] = 2 if negative_flag else 1
                board[y_cur, x_cur] = 1
              elif negative_flag and board_neg[y_cur, x_cur] < 1:
                board_neg[y_cur, x_cur] = 1
              y_cur += y_factor
          #print(dy, it)
          continue
        if dy == 0:
          if x_factor > 0:
            while x_cur <= x:
              if not negative_flag and board[y_cur, x_cur] < 1:
                #board[y_cur, x_cur] = 2 if negative_flag else 1
                board[y_cur, x_cur] = 1
              elif negative_flag and board_neg[y_cur, x_cur] < 1:
                board_neg[y_cur, x_cur] = 1
              x_cur += x_factor
          else:
            while x_cur >= x:
              if not negative_flag and board[y_cur, x_cur] < 1:
                #board[y_cur, x_cur] = 2 if negative_flag else 1
                board[y_cur, x_cur] = 1
              elif negative_flag and board_neg[y_cur, x_cur] < 1:
                board_neg[y_cur, x_cur] = 1
              x_cur += x_factor
          #print(dx, it)
          continue
        assert(dx != 0)
        grad = dy / dx
        assert(grad != 0.0)

        if abs(dy) > 1 or abs(dx) > 1:
          if abs(grad) > 1.0: # abs(dy) > abs(dx), steep
            for px in range(abs(dx)):
              if y_factor > 0:
                while y_cur <= (y_prev + ((x_cur-x_prev) * grad)):
                  if not negative_flag and board[y_cur, x_cur] < 1:
                    #board[y_cur, x_cur] = 2 if negative_flag else 1
                    board[y_cur, x_cur] = 1
                  elif negative_flag and board_neg[y_cur, x_cur] < 1:
                    board_neg[y_cur, x_cur] = 1
                  y_cur += y_factor
              else:
                while y_cur >= (y_prev + ((x_cur-x_prev) * grad)):
                  if not negative_flag and board[y_cur, x_cur] < 1:
                    #board[y_cur, x_cur] = 2 if negative_flag else 1
                    board[y_cur, x_cur] = 1
                  elif negative_flag and board_neg[y_cur, x_cur] < 1:
                    board_neg[y_cur, x_cur] = 1
                  y_cur += y_factor
              x_cur += x_factor
          elif abs(grad) < 1.0: # abs(dy) < abs(dx), gentle
            for py in range(abs(dy)):
              if x_factor > 0:
                while x_cur <= (x_prev + ((y_cur-y_prev) / grad)):
                  if not negative_flag and board[y_cur, x_cur] < 1:
                    #board[y_cur, x_cur] = 2 if negative_flag else 1
                    board[y_cur, x_cur] = 1
                  elif negative_flag and board_neg[y_cur, x_cur] < 1:
                    board_neg[y_cur, x_cur] = 1
                  x_cur += x_factor
              else:
                while x_cur >= (x_prev + ((y_cur-y_prev) / grad)):
                  if not negative_flag and board[y_cur, x_cur] < 1:
                    #board[y_cur, x_cur] = 2 if negative_flag else 1
                    board[y_cur, x_cur] = 1
                  elif negative_flag and board_neg[y_cur, x_cur] < 1:
                    board_neg[y_cur, x_cur] = 1
                  x_cur += x_factor
              y_cur += y_factor
  
          elif abs(dy) == abs(dx): # 45 deg
            if dy + dx != 0: # dy and dx have same sign
              for p in range(abs(dy)):
                if not negative_flag and board[y_cur, x_cur] < 1:
                  #board[y_cur, x_cur] = 2 if negative_flag else 1
                  board[y_cur, x_cur] = 1
                elif negative_flag and board_neg[y_cur, x_cur] < 1:
                  board_neg[y_cur, x_cur] = 1
                y_cur += y_factor
                x_cur += x_factor
            else: # dy and dx have opposite sign
              for p in range(abs(dy)):
                if not negative_flag and board[y_cur, x_cur] < 1:
                  #board[y_cur, x_cur] = 2 if negative_flag else 1
                  board[y_cur, x_cur] = 1
                elif negative_flag and board_neg[y_cur, x_cur] < 1:
                  board_neg[y_cur, x_cur] = 1
                y_cur += y_factor
                x_cur += x_factor
          else: # what?
            raise AssertionError
        else:
          #print('initially contacted case!')
          pass

    if id_num == 1:
      board1 = np.copy(board)
    elif id_num == 2:
      board2 = np.copy(board)  
  #y_min -= 2
  #y_max += 2
  #x_min -= 2
  #x_max += 2
  #board = board[y_min:y_max, x_min:x_max].astype(np.uint8)
  target_contour = {}
  target_contour[1] = board1
  target_contour[2] = board2
  target_contour['neg'] = board_neg
  #assert(y_min < np.inf and x_min < np.inf)
  #bbox = y_min, x_min, y_max, x_max
  return target_contour#, bbox

def fill_wrapper(contour, threshold, board, logical_not=False):
  if logical_not:
    board += np.logical_not(fill(contour>threshold)).astype(np.uint8)
  else:
    board += fill(contour>threshold).astype(np.uint8)

def contour2mask(contour, ret_key, ret_dict, lock=None):
    print('generating target mask..')
    mask = np.zeros(contour.shape, dtype=np.uint8)
    fill_wrapper(contour, 0, mask, ret_key == 'neg')
    
    # Use a lock to ensure thread safety when updating shared data
    if lock:
        with lock:
            ret_dict[ret_key] = mask.astype(np.uint8)
    else:
        ret_dict[ret_key] = mask.astype(np.uint8)
  

def xml2mask(fn, shape, div=None):
    shape_pad = (shape[0] + pad * 2, shape[1] + pad * 2)
    print('padding on the given shape: {} -> {}'.format(shape, shape_pad))
    contour = xml2contour(fn, shape_pad, div=div)  # dict
    threads = []
    target_mask = {}
    
    # Use a lock to ensure thread safety when updating shared data
    lock = Lock()

    for key, value in contour.items():
        if np.any(value) is None:
            continue
        t = Thread(target=contour2mask, args=(contour[key], key, target_mask, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if 2 in target_mask.keys():
        target_mask[2] = np.multiply(target_mask[2], target_mask['neg']).astype(np.uint8)
    print('unpadding: {} -> {}'.format(shape_pad, shape))
    for k in target_mask.keys():
        target_mask[k] = target_mask[k][pad:-pad, pad:-pad]
    del target_mask['neg']
    return target_mask  # dict


def glob_and_combine(directory_path, extensions=('svs', 'xml')):
    file_dict = {}

    for extension in extensions:
        # Use glob to find all files with the specified extension
        pattern = os.path.join(directory_path, f'*.{extension}')
        files = glob.glob(pattern)

        # Combine files with the same base name
        combined_files = {}
        for file_path in files:
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            combined_files.setdefault(base_name, []).append(file_path)

        # Add the combined files to the dictionary with the extension as the key
        file_dict[extension] = combined_files

    # Combine .svs and .xml files into tuples
    combined_list = []
    for base_name, svs_files in file_dict.get('svs', {}).items():
        xml_files = file_dict.get('xml', {}).get(base_name, [])
        combined_list.append((svs_files, xml_files))

    return combined_list
  
def glob_tif_files(directory_path):
    pattern = os.path.join(directory_path, '*.tif')
    tif_files = glob.glob(pattern)
    return [os.path.basename(tif_file) for tif_file in tif_files]
  
def parse_all_file(directory_path='./dataset/paip'):
  import pathlib
  mask_path = directory_path+"/mask/"
  result_list = glob_and_combine(directory_path)
  mask_files = glob_tif_files(mask_path)

  # Display the result
  for pairs in result_list:
      svs_files, xml_files = pairs
      wsi_fn = svs_files[0]
      xml_fn = xml_files[0]
      dst_fn = '{}_mask.tif'.format(pathlib.Path(xml_fn).name[:-4])
      if dst_fn in mask_files:
        print("Already got {} !".format(dst_fn))
        continue
      
      if not os.path.isfile(xml_fn):
        print('xml not found:', xml_fn)
        exit()
      if not os.path.isfile(wsi_fn):
        print('wsi not found:', wsi_fn)
        exit()

      print("We need to get {} !".format(dst_fn))
      src_shape = load_svs_shape(wsi_fn, level=0)
      dst_shape = load_svs_shape(wsi_fn, level=0)
      print(src_shape, '>>', dst_shape)

      mask = xml2mask(xml_fn, src_shape)
      for (k,v) in mask.items():
        mask[k] = v * (k*128-1) 
      
      print('saving mask img..')
      dst_path = os.path.join(mask_path, dst_fn)
      tifffile.imwrite(dst_path, sum(mask.values()))
      print('done!')

if __name__ == '__main__':
  import argparse, pathlib
  parser = argparse.ArgumentParser()
  parser.add_argument('--xml_fn', help='the filename of target XML annotation')
  parser.add_argument('--wsi_fn', help='the filename of reference WSI (for size inference)')
  parser.add_argument('--dst_fn', help='save tif filename')
  args = parser.parse_args()

  parse_all_file("/Volumes/dataset/paip")