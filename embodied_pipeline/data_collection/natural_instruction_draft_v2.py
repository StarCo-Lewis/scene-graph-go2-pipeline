"""Offline, review-only natural instruction draft; no API or robot calls."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import argparse, hashlib, json, random, re, shutil
from pathlib import Path
from .batch import read, read_jsonl, write, jsonl, sha, require_code_unchanged
from .catalog import _support_facts, _rooms, normalize_instruction

VERSION='natural_instruction_review_v2_20261024'
SCENES=('scene200','scene202','d3_619')
# Curated referring expressions, grounded in saved captions/edges; not human speech.
VOCAB={
 'scene200':{
  124:('the white geometric ornament on a black base','黑色底座上的白色几何摆件'),
  152:('the black computer mouse','黑色鼠标'),
  230:('the tape dispenser on the workbench','工作台上的胶带座'),
  182:('the dark grey storage tub with corner clips on its lid','盖子带角扣的深灰色收纳箱'),
  183:('the cardboard archive box with a finger-hole handle','带指孔提手的纸质档案盒'),
  184:('the light-coloured paper storage box on the black plastic crate','黑色塑料筐上的浅色纸质收纳盒'),
  224:('the flat document box on the narrow shelving unit','窄置物架上的扁平文件盒'),
  110:('the display pedestal supporting the white geometric ornament','放着白色几何摆件的展台'),
  133:('the long white table in the office','办公室里的白色长桌'),
  136:('the round black side table with three legs','三条腿的黑色圆边桌'),
  165:('the workbench in the workshop','工作间里的工作台'),
  1:('the white three-drawer filing cabinet','白色三抽屉文件柜'),
  163:('the grey metal rack with four shelves','四层灰色金属置物架'),
  167:('the narrow shelving unit supporting the flat document box','放着扁平文件盒的窄置物架'),
  180:('the open black plastic crate with lattice sides','侧面镂空的黑色塑料筐')},
 'scene202':{
  121:('the black bottle with the silver cap','银色盖子的黑色瓶子'),
  118:('the black notebook with an elastic band on the office A desk','办公室 A 桌上带松紧带的黑色笔记本'),
  123:('the black computer mouse','黑色鼠标'),
  289:('the black-and-red packing tape dispenser','黑红色封箱胶带座'),
  290:('the roll of white tape','白色胶带卷'),
  321:('the white mug on the cross-braced table','带交叉支撑桌腿的桌子上的白色杯子'),
  86:('the black hardcover book at the living-room coffee table','客厅茶几处的黑色硬皮书'),
  2:('the desk in office B','办公室 B 的桌子'),
  69:('the living-room coffee table','客厅茶几'),
  101:('the desk in office A','办公室 A 的桌子'),
  303:('the dark wood table with cross-braced legs','桌腿带交叉支撑的深色木桌'),
  307:('the writing desk with two drawers and a lower shelf','带两个抽屉和下层搁板的写字桌'),
  221:('the workbench supporting the white tape roll','放着白色胶带卷的工作台')},
 'd3_619':{
  32:('the open translucent container with ribbed sides','侧面带条纹的半透明敞口容器'),
  97:('the cardboard box with shipping labels and a logo','贴有运输标签和标志的纸箱'),
  176:('the tan rectangular pillow','棕褐色长方形靠枕'),
  180:('the square off-white pillow','米白色方形靠枕'),
  181:('the white rectangular pillow','白色长方形靠枕'),
  196:('the folded white cloth with a visible tag','带可见标签的白色折叠布料'),
  205:('the light grey blanket draped over the sofa corner','搭在沙发角上的浅灰色毯子'),
  301:('the cardboard box with a vertical strip of white tape','正面有竖向白胶带的纸箱'),
  179:('the round marble-patterned table with the orange slatted base','橙色条状底座的大理石纹圆桌'),
  46:('the light beige upholstered footstool','浅米色软包脚凳'),
  191:('the white modular sofa with a curved section','带弧形部分的白色组合沙发'),
  56:('the television on the wheeled tripod stand','带轮三脚支架上的电视'),
  13:('the dark office machine with a recessed front','正面内凹的深色办公设备'),
  11:('the transparent storage box with the curved top','顶部弯曲的透明收纳盒'),
  170:('the light beige sofa section with a vertical seam','带竖向接缝的浅米色沙发部分'),
  285:('the curved white upholstered partition','白色弧形软包隔挡'),
  182:('the light-coloured cushion under the white rectangular pillow','白色长方形靠枕下方的浅色坐垫')},
}
MOVERS={'scene200':[124,152,230,182,183,184,224],'scene202':[121,118,123,289,290,321,86],'d3_619':[32,97,176,180,181,196,205,301]}
SURFACES={'scene200':[110,133,136,165,1],'scene202':[2,69,101,303,307],'d3_619':[179,46,191]}
LANDMARKS={s:SURFACES[s]+{'scene200':[163],'scene202':[221],'d3_619':[56,13,11]}[s] for s in SCENES}
SOFT={176,180,181,196,205}
SOURCE={'scene200':{124:110,152:133,230:165,182:163,184:180,224:167},
 'scene202':{121:101,118:101,123:101,289:221,290:221,321:303},
 'd3_619':{176:170,180:285,181:182,205:191}}
RELATIVE={
 'scene200':{124:('the ornament on the display pedestal','展台上的摆件'),152:('the mouse on the white office table','办公室白桌上的鼠标'),230:VOCAB['scene200'][230],182:('the lidded tub on the four-shelf metal rack','四层金属架上的带盖收纳箱'),184:VOCAB['scene200'][184],224:VOCAB['scene200'][224]},
 'scene202':{121:('the bottle on the office A desk','办公室 A 桌上的瓶子'),118:VOCAB['scene202'][118],123:('the mouse on the office A desk','办公室 A 桌上的鼠标'),289:('the packing tape dispenser on the workbench with the white tape roll','放着白胶带卷的工作台上的封箱胶带座'),290:('the white tape roll on the workbench with the black-and-red dispenser','放着黑红色胶带座的工作台上的白色胶带卷'),321:VOCAB['scene202'][321]},
 'd3_619':{176:('the tan pillow resting on the light beige sofa section','浅米色沙发部分上的棕褐色靠枕'),180:('the square pillow resting on the curved upholstered partition','弧形软包隔挡上的方形靠枕'),181:('the white rectangular pillow resting on another cushion','另一块坐垫上的白色长方形靠枕'),205:VOCAB['d3_619'][205]}}
FAMILIES={'navigate':6,'single_transport':25,'hold_at':20,'protected_transport':30,'two_objectives':32,'three_objectives':20,'ordered_visits':20,'delivery_and_wait':27,'clarification':20}
STYLES={'explicit_id':20,'descriptive':40,'relational':30,'purpose_led':40,'contextual':50,'clarification':20}
STYLE_ZH={'explicit_id':'显式 ID','descriptive':'属性描述／口语请求','relational':'空间关系指称','purpose_led':'目的与约束','contextual':'带前文的指代／修正','clarification':'信息不足，应先澄清'}
FAMILY_ZH={'navigate':'单点导航','single_transport':'单物搬运','hold_at':'送达后持有','protected_transport':'保护要求','two_objectives':'双目标','three_objectives':'三目标','ordered_visits':'有序访问','delivery_and_wait':'搬运后到指定处待命','clarification':'澄清'}

# Compact everyday names retain useful distinctions without reciting captions.
SHORT={
 'scene200':{124:('the white geometric ornament','白色几何摆件'),152:('the mouse','鼠标'),
  182:('the dark grey lidded tub','深灰色带盖收纳箱'),183:('the archive box with the finger hole','带指孔的档案盒'),
  110:('the pedestal with the white ornament','放着白色摆件的展台'),133:('the office table','办公室桌子'),
  136:('the black round side table','黑色圆边桌'),165:('the workshop bench','工作间工作台'),1:('the white filing cabinet','白色文件柜')},
 'scene202':{121:('the silver-capped bottle','银盖瓶子'),118:('the black notebook on the office A desk','办公室 A 桌上的黑色笔记本'),
  123:('the mouse','鼠标'),289:('the black-and-red tape dispenser','黑红色胶带座'),290:('the white tape roll','白色胶带卷'),
  2:('the office B desk','办公室 B 的桌子'),101:('the office A desk','办公室 A 的桌子'),
  307:('the two-drawer writing desk','双抽屉写字桌'),303:('the cross-braced wooden table','桌腿带交叉支撑的木桌')},
 'd3_619':{32:('the ribbed translucent container','带条纹的半透明容器'),97:('the shipping-labelled cardboard box','贴运输标签的纸箱'),
  176:('the tan pillow','棕褐色靠枕'),180:('the square cream pillow','米白色方形靠枕'),181:('the rectangular white pillow','白色长方形靠枕'),
  196:('the folded cloth with the tag','带标签的折叠布料'),205:('the grey blanket','灰色毯子'),301:('the box with the vertical white tape','贴竖向白胶带的箱子'),
  179:('the round table with the orange base','橙色底座的圆桌'),46:('the beige footstool','米色脚凳'),191:('the curved white modular sofa','带弧形部分的白色组合沙发'),
  56:('the TV on the wheeled stand','带轮支架上的电视'),13:('the dark office machine','深色办公设备')},
}

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def label(scene,oid,style):
    if style=='explicit_id':
        en,zh=VOCAB[scene][oid]
        return en+' (ID '+str(oid)+')',zh+'（ID '+str(oid)+'）'
    if style=='relational':return RELATIVE[scene].get(oid,SHORT[scene].get(oid,VOCAB[scene][oid]))
    return SHORT[scene].get(oid,VOCAB[scene][oid])

def render_pair(pair,labels):
    en={k:v[0] for k,v in labels.items()};zh={k:v[1] for k,v in labels.items()}
    for k in ('a','b'):
        en[k.upper()]=en.get(k,'')[:1].upper()+en.get(k,'')[1:];zh[k.upper()]=zh.get(k,'')
    return pair[0].format(**en),pair[1].format(**zh)

def allowed(scene,o,t,facts):
    if o==t or t in {x['support_id'] for x in facts.get(o,[])}:return False
    return not(scene=='d3_619' and t in (46,191) and o not in SOFT)

def ops_for(scene,family,style,rng,facts):
    a=rng.choice(list(RELATIVE[scene]) if style=='relational' else MOVERS[scene])
    b,c=rng.sample([x for x in MOVERS[scene] if x!=a],2)
    d,e,f=rng.sample(SURFACES[scene],3)
    p=rng.choice([x for x in MOVERS[scene] if x not in (a,b,c)]);ordered=True
    if family=='navigate':
        a=rng.choice(LANDMARKS[scene]);ops=[('visit',a,None)];p=None
    elif family in ('single_transport','protected_transport'):
        ops=[('deliver',a,d)];p=p if family=='protected_transport' else None
    elif family=='hold_at':
        d=rng.choice(LANDMARKS[scene]);ops=[('hold',a,d)];p=None
    elif family in ('two_objectives','three_objectives'):
        ordered=rng.choice([True,False]);ops=[('deliver',a,d),('deliver',b,e)];p=None
        if family=='three_objectives':ops.append(('deliver',c,f))
    elif family=='ordered_visits':
        a,b,c=rng.sample(LANDMARKS[scene],3);ops=[('visit',a,None),('visit',b,None),('visit',c,None)];p=None
    elif family=='delivery_and_wait':
        e=rng.choice([x for x in LANDMARKS[scene] if x!=d]);ops=[('deliver',a,d),('visit',e,None)];p=None
    else:raise ValueError(family)
    if any(mode=='deliver' and not allowed(scene,o,t,facts) for mode,o,t in ops):return None
    return ops,p,ordered,dict(a=a,b=b,c=c,d=d,e=e,f=f,p=p)

BANKS={
 'navigate':[
  ('Come over to {a} and stop nearby.','到{a}附近停下。'),
  ('Could you wait beside {a}?','去{a}旁边等着。'),
  ('Go over to {a}; leave everything where it is.','到{a}那里，不要挪任何东西。'),
  ('Your next stop is {a}. Just get close enough to be beside it.','下一站是{a}，只要到旁边。'),
  ('Head for {a}, but this is only a visit, not a pickup.','去{a}那里，只过去，不拿东西。'),
  ('Please park yourself near {a} for now.','暂时在{a}附近停下。'),
  ('Can you get over to {a} without moving any objects?','到{a}那里，不要移动物品。'),
  ('I only need you beside {a}; nothing needs carrying.','只需要你到{a}旁边，不需要搬东西。')],
 'single_transport':[
  ('Could you put {a} on {d}?','把{a}放到{d}上。'),
  ('Move {a} over to {d} and leave it on top.','把{a}移到{d}上，留在那里。'),
  ('I would like {a} on {d}, please.','我想把{a}放在{d}上。'),
  ('Set {a} down on {d} when you get there.','把{a}带到{d}，放在上面。'),
  ('{A} needs to go on {d}. Can you take care of that?','把{a}放到{d}上。'),
  ('Take {a} across to {d}; you can put it down on top.','把{a}送到{d}，放在上面。'),
  ('Leave {a} on {d} rather than carrying it around.','把{a}放在{d}上，不要一直拿着。'),
  ('Please relocate {a} to {d}. It should be resting on the top when you finish.','把{a}搬到{d}上，结束时放稳。')],
 'hold_at':[
  ('Bring {a} over to {d}, but keep hold of it when you arrive.','把{a}带到{d}附近，到达后继续拿着。'),
  ('Take {a} to {d}. Do not put it down yet.','把{a}送到{d}那里，先别放下。'),
  ('Pick up {a} and wait with it beside {d}.','拿起{a}，到{d}旁边拿着等候。'),
  ('I have not decided where to put {a}. For now, hold it near {d}.','我还没决定{a}放哪里，先拿着到{d}附近。'),
  ('Get {a} over to {d} and keep it in your grasp.','把{a}带到{d}附近，继续抓持。'),
  ('Do not set {a} on anything. Carry it as far as {d} and stop there.','不要把{a}放在任何东西上，拿到{d}附近停下。')],
 'protected_transport':[
  ('Put {a} on {d}, but leave {p} exactly where it is.','把{a}放到{d}上，{p}留在原处。'),
  ('Only {a} should move to {d}; put it down on top. {P} is staying put.','只把{a}移到{d}上放下，{p}不要动。'),
  ('Could you shift {a} onto {d} without disturbing {p}?','把{a}移到{d}上，别动{p}。'),
  ('Leave {p} alone. It is {a} that I want on {d}.','别动{p}，我要搬到{d}上的是{a}。'),
  ('Take {a} to {d} and set it down on top. Do not pick up {p}, even temporarily.','把{a}放到{d}上；{p}连临时拿起也不行。'),
  ('Please move {a}, not {p}, onto {d}.','要移到{d}上的是{a}，不是{p}。'),
  ('Make this one change: {a} goes on {d}. Keep {p} untouched throughout.','只改动这一点：把{a}放在{d}上，全程别动{p}。')],
 'two_ordered':[
  ('Set {a} on {d} first. After that, leave {b} on {e}.','先把{a}放到{d}上，再把{b}放到{e}上。'),
  ('Before dealing with {b}, put {a} on {d}. {B} should then go on {e}.','先把{a}放在{d}上，再处理{b}，把它放到{e}上。'),
  ('Start with {a}: leave it on {d}. Finish by putting {b} on {e}.','先把{a}放在{d}上，最后把{b}放到{e}上。'),
  ('I need {a} on {d} before you take {b} over to {e} and put it down on top.','先把{a}放到{d}上，再把{b}放到{e}上。')],
 'two_unordered':[
  ('Put {a} on {d} and {b} on {e}. Either one can go first.','把{a}放到{d}上，把{b}放到{e}上，先做哪个都行。'),
  ('Here is the arrangement I want: {a} on {d}, and {b} on {e}. Choose the order yourself.','想要的摆放是：{a}在{d}上，{b}在{e}上，顺序你定。'),
  ('I have two things to relocate, with no preferred order: {a} onto {d}, and {b} onto {e}. Leave both resting there.','需要挪两样东西：{a}放到{d}上，{b}放到{e}上，不限定先后。'),
  ('Could you leave {a} on {d} and {b} on {e}? I do not mind which you do first.','把{a}放在{d}上、{b}放在{e}上即可，不限顺序。')],
 'three_ordered':[
  ('First leave {a} on {d}, then put {b} on {e}. Only after those are done, set {c} on {f}.','先把{a}放到{d}，再把{b}放到{e}，这两件做完才把{c}放到{f}。'),
  ('Do the rearrangement in this order: {a} onto {d}, {b} onto {e}, and finally {c} onto {f}.','按此顺序调整：{a}放到{d}、{b}放到{e}、最后{c}放到{f}。'),
  ('I need {a} on {d} before {b} goes on {e}; leave {c} on {f} last.','先把{a}放到{d}，然后{b}放到{e}，最后{c}放到{f}。')],
 'three_unordered':[
  ('Help me set things out: {a} goes on {d}, {b} on {e}, and {c} on {f}. Any order is fine.','帮我摆好：{a}放到{d}、{b}放到{e}、{c}放到{f}，顺序不限。'),
  ('When you finish, I want {a} resting on {d}, {b} on {e}, and {c} on {f}. Work through them in whichever order you prefer.','结束时要{a}在{d}上、{b}在{e}上、{c}在{f}上，处理顺序你定。'),
  ('Three things need moving: put {a} on {d}, {b} on {e}, and {c} on {f}. There is no priority between them.','有三样要挪：{a}到{d}、{b}到{e}、{c}到{f}，均放下，不分优先级。')],
 'ordered_visits':[
  ('Stop by {a}, then {b}, and finish beside {c}. Do not move any objects.','依次到{a}、{b}，最后在{c}旁边停下，不挪物品。'),
  ('I want you to visit {a} before {b}, and leave {c} until last. This is a walk-through only.','先到{a}，再到{b}，最后到{c}，只走访。'),
  ('Make {a} your first stop and {c} your last, with a stop at {b} in between. Leave the objects alone.','先到{a}，中间到{b}，最后到{c}，不碰物品。'),
  ('Go to {a}. Once you have been there, go over to {b}; end the request at {c}. Nothing needs picking up.','先到{a}，再去{b}，最后停在{c}，不抓取东西。')],
 'delivery_and_wait':[
  ('Leave {a} on {d}, then wait beside {e} for the next request.','把{a}放到{d}上，然后到{e}旁边等下一条指令。'),
  ('Once {a} is resting on {d}, I need you over at {e}. Leave the item behind.','{a}放好在{d}上以后，你到{e}那里，不要再带着物品。'),
  ('Drop {a} off on {d} by placing it down gently. Your waiting spot afterward is {e}.','把{a}轻放在{d}上，之后到{e}等候。'),
  ('Put {a} onto {d}. Do not stay there afterward; finish beside {e}.','把{a}放到{d}上，然后不要留在那里，最后到{e}旁边。')],
}
PURPOSE={
 'navigate':[
  ('I need the robot ready at {a}. Go there and wait without touching the objects.','我需要机器人到{a}待命，过去等候，不碰物品。'),
  ('We will start from {a} later. For now, just move beside it.','稍后从{a}那里开始，现在先到旁边。')],
 'single_transport':[
  ('I want {a} out of its current spot. Use {d} as its temporary resting place.','我想挪开{a}，临时放到{d}上。'),
  ('For the next part of my work, I need {a} waiting on {d}.','接下来要用{a}，先把它放在{d}上。'),
  ('Help me rearrange things: {a} should end up on {d}.','帮我调整一下摆放，把{a}放到{d}上。'),
  ('I am changing where I work. Put {a} on {d} for me.','我准备换个工作位置，把{a}放到{d}上。')],
 'hold_at':[
  ('I need {a} moved, but I am not ready to set it down. Bring it to {d} and keep holding it.','我需要移走{a}，但还不准备放下；带到{d}附近，继续拿着。'),
  ('Before I choose a resting place for {a}, carry it to {d} and wait with it in your grasp.','我还没选好{a}的放置处，先拿到{d}附近等候。')],
 'protected_transport':[
  ('I am rearranging just {a}, not {p}. Put it on {d} and leave {p} alone.','只调整{a}，放到{d}上；{p}别动。'),
  ('I want {a} out of its current spot, but {p} must stay undisturbed. Use {d} for {a}.','把{a}挪到{d}上，{p}始终保持不动。'),
  ('Help me make room by moving {a} onto {d}. Do not move {p} as part of the rearrangement.','帮我腾位置，把{a}搬到{d}上，别挪{p}。')],
}

def compose(scene,family,style,index,ops,p,ordered,ids):
    labels={k:label(scene,v,style) for k,v in ids.items() if v is not None}
    key=family
    if family in ('two_objectives','three_objectives'):key=family.split('_')[0]+('_ordered' if ordered else '_unordered')
    bank=PURPOSE.get(family,BANKS[key]) if style=='purpose_led' else BANKS[key]
    pair=bank[index%len(bank)]
    if style=='purpose_led' and family not in PURPOSE:
        pair=('Help me prepare for the next part of my work. '+pair[0],'帮我为接下来的工作做准备。'+pair[1])
    context=[];context_zh=[]
    if style=='contextual':
        if index%3==0:
            focus='a' if family in ('navigate','ordered_visits') else 'd'
            named=labels[focus]
            context=[{'role':'user','content':'For this request, when I say "there", I mean '+named[0]+'.'}]
            context_zh=['这次我说“那里”，指的是'+named[1]+'。']
            labels[focus]=('the place I just named','我刚才说的地方')
        elif index%3==1:
            context=[{'role':'user','content':'The item I mean is '+labels['a'][0]+'. I have not asked you to move it yet.'}]
            context_zh=['我说的是'+labels['a'][1]+'，我还没让你开始移动。']
            labels['a']=('that same item','刚才说的那个物体')
        else:
            context=[{'role':'user','content':'I was considering '+labels['b'][0]+', but do not start anything yet.'}]
            context_zh=['我刚才在考虑'+labels['b'][1]+'，但先不要开始行动。']
            pair=('Change of plan. '+pair[0],'改一下计划。'+pair[1])
    # Uppercase slot without lowercasing object IDs or room aliases.
    if 'p' in labels:labels['P']=(labels['p'][0][:1].upper()+labels['p'][0][1:],labels['p'][1])
    en,zh=render_pair(pair,labels)
    if style=='contextual':
        en=en.replace('onto the place I just named','there').replace('on the place I just named','there')
    return en,zh,context,context_zh,key+'_'+str(index%len(bank))

AMBIGUOUS=[
 ('Take that one over there.','把那个拿到那边去。',['referent','destination']),
 ('Put it back where it belongs.','把它放回该放的地方。',['referent','home_location']),
 ('Bring me the thing I used earlier.','把我之前用的那个东西拿来。',['prior_use_history','user_pose']),
 ('Move the nearest one out of the way.','把最近的那个挪开。',['object_class','distance_reference','destination']),
 ('Clear this spot for me.','帮我把这里腾出来。',['user_pointing_region','scope','destination']),
 ('Put the spare one on {d}.','把备用的那个放到{d}上。',['spare_identity']),
 ('Bring my usual one over to {d}.','把我常用的那个拿到{d}那里。',['ownership_or_preference','release_or_hold']),
 ('Move the one on the left onto {d}.','把左边的那个放到{d}上。',['object_class','observer_frame']),
 ('Take {a} back to where it was yesterday.','把{a}放回昨天的位置。',['historical_location']),
 ('Put {a} somewhere better.','把{a}放到更合适的地方。',['placement_preference']),
 ('Bring {a} to me.','把{a}拿给我。',['user_pose','handover_capability']),
 ('Move the broken one onto {d}.','把坏的那个移到{d}上。',['functional_condition']),
 ('Bring the clean one to {d} and put it down.','把干净的那个拿到{d}放下。',['cleanliness','object_class']),
 ('Put {a} on my desk.','把{a}放到我的桌子上。',['desk_ownership']),
 ('Move everything I do not need off {d}.','把{d}上我不需要的东西都挪走。',['user_relevance','destination']),
 ('Leave {a} beside the other one.','把{a}放在另一个旁边。',['other_referent','placement_surface']),
 ('Use the same place as last time for {a}.','{a}还放上次那个地方。',['previous_task_history']),
 ('Put {a} in the room next door.','把{a}放到隔壁房间。',['current_room','room_adjacency','placement_surface']),
 ('Move {a} closer, but not too close.','把{a}挪近些，但别太近。',['distance_reference','desired_distance','placement_surface']),
 ('Tidy up {d} the way I like it.','按我喜欢的样子整理{d}。',['user_preference','scope','destination']),
]

def reference(ops,protect,ordered,family,en):
    steps=[];last={};holds=None
    for i,(mode,o,t) in enumerate(ops,1):
        steps.append({'step_id':'r'+str(i).zfill(2),'intent':mode,'object_id':o,'destination_id':t,
                      'depends_on':[] if i==1 or not ordered else ['r'+str(i-1).zfill(2)]})
        if mode=='deliver':last[o]=t;holds=False
        elif mode=='hold':holds=o
    final=[{'predicate':'on','object_id':o,'support_id':t} for o,t in last.items()]
    final += [{'predicate':'held','object_id':o,'expected':False} for o in last]
    if holds is not False and holds is not None:
        final.extend([{'predicate':'held','object_id':holds,'expected':True},{'predicate':'near','object_id':ops[-1][2]}])
    if ops[-1][0]=='visit':final.append({'predicate':'near','object_id':ops[-1][1]})
    return {'expected_response':'plan_or_evidence_based_clarification','steps':steps,'final_predicates':final,
            'process_constraints':{'preserve_objects':[] if protect is None else [protect],
              'no_manipulation_intent':family in ('navigate','ordered_visits'),
              'explicit_no_manipulation_prohibition':bool(re.search('without touching|without moving|nothing needs|not a pickup|not move any|leave everything|objects alone|nothing needs carrying',en,re.I))},
            'ordering':'requested_order' if ordered else 'any_dependency_safe_order',
            'reference_source':'rule_authored_from_same_intent_not_independent_oracle',
            'review_status':'pending_user_review','physical_feasibility':'unverified',
            'unknowns':['robot_pose','held_state','safe_approach','payload','collision_free_path','surface_free_space'],
            'scoring_note':'Natural referring expressions still need uniqueness review. If the supplied graph cannot disambiguate, clarification is permitted; do not force the intended ID.'}

def record(scene,rid,family,style,en,zh,context,context_zh,ref,timestamp,by,rooms,room_provenance,base):
    ids=ref['grounded_object_ids'];support_ids={x['support_id'] for x in ref['source_support_hypotheses']}
    evidence=[{'object_id':i,'object_tag':by[i]['object_tag'],'caption':by[i].get('caption'),'room':rooms.get(i),
               'source':'supplied_graph_caption_or_existing_author_room_annotation_not_physical_truth'} for i in sorted(set(ids)|support_ids)]
    labels={rooms[i] for i in ids if i in rooms}
    room_scope='unknown' if any(i not in rooms for i in ids) or not ids else 'cross_room' if len(labels)>1 else 'single_room'
    return {'record_id':rid,'scene_id':scene,'draft_version':VERSION,'created_at_utc':timestamp,
      'source_scene_capture_time':None,'source_scene_capture_time_note':'Not inferred from graph timestamp; consult original modality provenance.',
      'graph_version':'sha256:'+sha(base/'graph/scene_graph.json'),'edges_sha256':sha(base/'graph/scene_graph_edges.json'),
      'instruction_language':'en','instruction':en,'reading_aid_zh':zh,'context_reading_aid_zh':context_zh,
      'expression_style':style,'task_family':family,'horizon':'clarification' if family=='clarification' else 'long' if len(ref['steps'])>1 else 'short',
      'room_scope':room_scope,'room_annotation_provenance':room_provenance,
      'model_input':{'context_messages':context,'instruction':en,
         'context_status':'authored_user_preface_not_robot_execution_history' if context else 'none',
         'scene_graph':'cases/'+scene+'/graph/scene_graph.json','scene_graph_edges':'cases/'+scene+'/graph/scene_graph_edges.json'},
      'reference_artifact':'reference.review_only.json','reviewer_only_graph_evidence':evidence,
      'review_status':'pending_user_review','collection_status':'not_run','raw_model_response':None,
      'source_type':'synthetic_graph_grounded_natural_language_draft_not_collected_human_speech',
      'valid_candidate':None,'training_eligible':False,'physical_execution_verified':False,
      'review_checks_required':['naturalness','referent_uniqueness','task_usefulness','graph_conflicts','feasibility','near_duplicates'],
      'modality_resource_index':str(base/'modality_index.json'),
      'available_skills_reference':str(base.parents[1]/'shared/actual_collection_skill_library.json'),
      'source_permissions':'inherited_unknown_pending_review','instruction_sha256':digest([context,en])}

def build(bundle,dest):
    if dest.exists():raise FileExistsError('Use a fresh directory; never overwrite reviewed drafts.')
    require_code_unchanged(bundle);dest.mkdir(parents=True)
    all_rows=[];all_refs=[];counts={};timestamp=datetime.now(timezone.utc).isoformat()
    for si,scene in enumerate(SCENES):
        base=bundle/'cases'/scene;by={n['id']:n for n in read(base/'graph/scene_graph.json')}
        facts=_support_facts(read(base/'graph/scene_graph_edges.json'));rooms,rprov=_rooms(scene,by)
        for o,s in SOURCE[scene].items():assert any(x['support_id']==s for x in facts.get(o,[])),(scene,o,s)
        for oid in VOCAB[scene]:assert oid in by
        out=dest/'cases'/scene;(out/'graph').mkdir(parents=True)
        for name in ('scene_graph.json','scene_graph_edges.json'):shutil.copy2(base/'graph'/name,out/'graph'/name)
        shutil.copy2(base/'modality_index.json',out/'source_modality_index.json')
        styles=[s for s,n in STYLES.items() if s!='clarification' for _ in range(n)]
        scene_families=dict(FAMILIES)
        # The real graph has few defensible complete placement surfaces. Avoid
        # inventing surfaces or padding single transports with paraphrases.
        if scene=='d3_619':
            scene_families['single_transport']=10
            scene_families['delivery_and_wait']=42
        families=[f for f,n in scene_families.items() if f!='clarification' for _ in range(n)]
        rng=random.Random(20261024+si);rng.shuffle(styles);rng.shuffle(families)
        # A relation-labelled record must actually describe an object by a source
        # relationship, not merely be an attribute-only landmark navigation.
        for i,fam in enumerate(families):
            if fam in ('navigate','ordered_visits') and styles[i]=='relational':
                j=next(j for j in range(len(styles)) if families[j] not in ('navigate','ordered_visits') and styles[j]!='relational')
                styles[i],styles[j]=styles[j],styles[i]
        old_sigs=set()
        for t in read_jsonl(base/'tasks.jsonl'):
            sig=[('visit' if m['predicate']=='visited_near' else 'hold' if m['predicate']=='transported_while_held' else 'deliver',m['object_id'],m.get('destination_id')) for m in t['goal']['ordered_milestones']]
            old_sigs.add(digest([sig,t['constraints'].get('preserve_objects',[])]))
        rows=[];refs=[];seen=set();text_seen=set()
        jobs=list(zip(families,styles));rng.shuffle(jobs)
        for i,(fam,style) in enumerate(jobs,1):
            for attempt in range(10000):
                candidate=ops_for(scene,fam,style,rng,facts)
                if candidate is None:continue
                ops,p,ordered,ids=candidate
                # Canonicalize independent goal sets so order variants do not
                # masquerade as unique semantics when the user allows any order.
                sig_ops=ops if ordered else sorted(ops)
                semantic=digest([sig_ops,p,ordered])
                if semantic in seen:continue
                en,zh,ctx,cz,pattern=compose(scene,fam,style,i+attempt,ops,p,ordered,ids)
                key=normalize_instruction(json.dumps(ctx,ensure_ascii=False)+' '+en)
                if key in text_seen:continue
                break
            else:raise RuntimeError(('Cannot meet design',scene,fam,style))
            seen.add(semantic);text_seen.add(key)
            rid=scene+'_'+str(10000000+i);ref=reference(ops,p,ordered,fam,en)
            used=sorted({v for mode,o,t in ops for v in (o,t) if v is not None}|({p} if p else set()))
            ref.update({'record_id':rid,'grounded_object_ids':used,'semantic_signature':semantic,
                'overlaps_old_objective_signature':digest([ops,[] if p is None else [p]]) in old_sigs,
                'source_support_hypotheses':[x for o in used for x in facts.get(o,[])]})
            r=record(scene,rid,fam,style,en,zh,ctx,cz,ref,timestamp,by,rooms,rprov,base)
            r['surface_pattern_id']=pattern;rows.append(r);refs.append(ref)
        for j,(en,zh,gaps) in enumerate(AMBIGUOUS,181):
            a=MOVERS[scene][(j-181)%len(MOVERS[scene])];d=SURFACES[scene][(j-181)%len(SURFACES[scene])]
            mentioned=([a] if '{a}' in en else [])+([d] if '{d}' in en else [])
            en,zh=render_pair((en,zh),{'a':VOCAB[scene][a],'d':VOCAB[scene][d]});rid=scene+'_'+str(10000000+j)
            ref={'record_id':rid,'expected_response':'clarify_before_action','missing_information':gaps,
              'acceptable_behavior':'Ask targeted questions; preserve uncertainty; do not choose an arbitrary target or invent history.',
              'steps':[],'final_predicates':[],'grounded_object_ids':mentioned,'semantic_signature':digest(['clarify',gaps,mentioned]),
              'hidden_target':None,'review_status':'pending_user_review','reference_source':'authored_information_gap_not_model_answer',
              'overlaps_old_objective_signature':False,'source_support_hypotheses':[]}
            rows.append(record(scene,rid,'clarification','clarification',en,zh,[],[],ref,timestamp,by,rooms,rprov,base));refs.append(ref)
        jsonl(out/'tasks.jsonl',rows);jsonl(out/'references.review_only.jsonl',refs)
        for r,ref in zip(rows,refs):
            write(out/'tasks'/r['record_id']/'task.json',r)
            write(out/'tasks'/r['record_id']/'reference.review_only.json',ref)
            write(out/'tasks'/r['record_id']/'model_input.json',r['model_input'])
        counts[scene]={'records':len(rows),'expression_styles':dict(Counter(r['expression_style'] for r in rows)),
          'task_families':dict(Counter(r['task_family'] for r in rows)),'horizons':dict(Counter(r['horizon'] for r in rows)),
          'with_id_in_user_payload':sum(bool(re.search(r'\bID\s*\d+',json.dumps(r['model_input']))) for r in rows),
          'with_context':sum(bool(r['model_input']['context_messages']) for r in rows),
          'distinct_intent_signatures':len({x['semantic_signature'] for x in refs}),
          'overlap_old_intent_records':sum(x['overlaps_old_objective_signature'] for x in refs)}
        assert len(rows)==200 and Counter(r['expression_style'] for r in rows)==Counter(STYLES)
        assert counts[scene]['with_id_in_user_payload']==20
        render(out/'instructions_review.html',rows,refs)
        (out/'instructions_only.txt').write_text(instruction_text(rows),encoding='utf-8')
        all_rows+=rows;all_refs+=refs
    jsonl(dest/'dataset.review_only.jsonl',all_rows);jsonl(dest/'references.review_only.jsonl',all_refs)
    jsonl(dest/'model_inputs.preview.jsonl',[{'record_id':r['record_id'],'scene_id':r['scene_id'],**r['model_input']} for r in all_rows])
    render(dest/'instructions_review.html',all_rows,all_refs)
    (dest/'instructions_only.txt').write_text(instruction_text(all_rows),encoding='utf-8')
    summary={'version':VERSION,'created_at_utc':timestamp,'status':'draft_pending_user_approval_not_run',
      'model_calls':0,'robot_calls':0,'total_records':600,'scene_counts':counts,'approval':None,'source_bundle':str(bundle),
      'limits':['Synthetic authored instructions, not collected human speech.',
       'Context is a supplied user preface, never fabricated completed robot execution.',
       'Reviewer references and intended IDs must not be supplied to the model.',
       'Clarification is expected for information-gap cases; these do not require an engineering JSON before clarification.',
       'All naturalness, referent uniqueness, physical affordances and task usefulness remain pending user review.',
       'Graph anomalies remain unchanged. D3-619 room membership unknown; no room topology used.',
       'Task records may share an objective; near-duplicate groups are separate and not automatically valid unique samples.',
       'Only navigation is available on the current real robot. Manipulation cases are offline reasoning candidates.',
       'No launch script or collection run until user approval.']}
    write(dest/'draft_summary.json',summary)
    groups={}
    for r,ref in zip(all_rows,all_refs):groups.setdefault((r['scene_id'],ref['semantic_signature']),[]).append(r['record_id'])
    write(dest/'near_duplicate_review.json',{'groups':[{'scene_id':k[0],'semantic_signature':k[1],'record_ids':v} for k,v in groups.items() if len(v)>1],
       'policy':'Language variants remain visible; not automatically independent valid semantic tasks.'})
    shutil.copy2(__file__,dest/'generation_script.py')
    write(dest/'generation_config.json',{'version':VERSION,'seed':20261024,'styles':STYLES,'families':FAMILIES,'vocabulary':VOCAB,'source_relations':SOURCE})
    write(dest/'file_manifest.json',[{'path':str(p.relative_to(dest)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file()])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def instruction_text(rows):
    lines=['REVIEW DRAFT - NOT RUN. English is the planned input; Chinese is a reading aid.','']
    for r in rows:
        lines.append(r['record_id']+' | '+STYLE_ZH[r['expression_style']]+' | '+FAMILY_ZH[r['task_family']])
        for c,z in zip(r['model_input']['context_messages'],r['context_reading_aid_zh']):lines.extend(['Previous user: '+c['content'],'前文：'+z])
        lines.extend(['User: '+r['instruction'],'中文辅助：'+r['reading_aid_zh'],''])
    return '\n'.join(lines)

def render(path,rows,refs):
    h=escape
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自然指令第二版 · 待审阅</title>',
      '<style>body{max-width:1200px;margin:28px auto;padding:0 20px;font:16px/1.65 system-ui;color:#20334a;background:#f7f9fc}h1,h2{color:#174568}.bar{position:sticky;top:0;background:#e5edf6;padding:12px;z-index:2}input,select{padding:9px;margin:4px;max-width:90%}article{background:white;border:1px solid #d5dfeb;border-radius:8px;margin:14px 0;padding:16px}small,.zh{color:#55677b}.tag{color:#245d8c}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f8;padding:12px;font-size:12px}.context{border-left:3px solid #7691ad;padding:8px 12px;background:#f4f7fa}.warn{color:#9d3e25}summary{cursor:pointer}article[hidden]{display:none}</style>',
      '<h1>Scene200 / Scene202 / D3-619：自然指令审阅稿</h1>',
      '<p>三个场景各200条，英文是拟输入模型的指令，中文仅作阅读辅助。全部尚未运行。带前文的条目必须连同前文输入；图和技能库另行提供，参考答案不得输入模型。</p>',
      '<p class="warn">每场景20条显式ID、40条属性／口语描述、30条关系指称、40条目的表达、50条带上下文、20条应澄清。采用人工设计表达与任务组合，不是采集的人类原话。指称唯一性、用途合理性与图关系仍待审阅；D3-619不虚构房间。</p>',
      '<p>已检查批内结构化任务签名并保留近重复检查文件；这不替代人工语义去重。与旧批次目标重合的条目在参考信息中标明。60条信息不足请求预期先澄清，不强行要求工程JSON。其余540条也允许对确有依据的图歧义进行澄清。当前机器人只有导航能力，操作类仅作离线推理。</p>',
      '<div class="bar"><input id="query" placeholder="搜索编号、指令、中文关键词" size="42"><select id="scene"><option value="">所有场景</option>'+''.join('<option>'+s+'</option>' for s in SCENES)+'</select><select id="style"><option value="">所有表达</option>'+''.join('<option value="'+k+'">'+v+'</option>' for k,v in STYLE_ZH.items())+'</select><span id="count"></span></div>']
    for r,ref in zip(rows,refs):
        parts.append('<article data-scene="'+r['scene_id']+'" data-style="'+r['expression_style']+'"><h2>'+r['record_id']+'</h2><div class="tag">'+STYLE_ZH[r['expression_style']]+' · '+FAMILY_ZH[r['task_family']]+' · '+r['horizon']+'</div>')
        for c,z in zip(r['model_input']['context_messages'],r['context_reading_aid_zh']):parts.append('<div class="context"><b>需一同输入的前文：</b><br>'+h(c['content'])+'<br><small>'+h(z)+'</small></div>')
        parts.append('<p><b>当前指令：</b> '+h(r['instruction'])+'</p><p class="zh">'+h(r['reading_aid_zh'])+'</p>')
        parts.append('<details><summary>审阅用预期与图证据（不输入模型）</summary><pre>'+h(json.dumps({'reference':ref,'graph_evidence':r['reviewer_only_graph_evidence']},ensure_ascii=False,indent=2))+'</pre></details></article>')
    parts.append('''<script>const cards=[...document.querySelectorAll('article')];function filter(){const q=document.getElementById('query').value.toLowerCase(),s=document.getElementById('scene').value,t=document.getElementById('style').value;let n=0;cards.forEach(c=>{c.hidden=!!((s&&c.dataset.scene!==s)||(t&&c.dataset.style!==t)||(q&&!c.textContent.toLowerCase().includes(q)));if(!c.hidden)n++});document.getElementById('count').textContent='显示 '+n+' / '+cards.length+' 条'}['query','scene','style'].forEach(id=>document.getElementById(id).addEventListener('input',filter));filter();</script></html>''')
    path.write_text('\n'.join(parts),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.bundle,a.output)
