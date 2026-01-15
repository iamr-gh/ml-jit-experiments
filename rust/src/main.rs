use std::collections::HashMap;

#[derive(Debug)]
pub enum Op {
    Add(Box<Op>, Box<Op>),
    Sub(Box<Op>, Box<Op>),
    Mul(Box<Op>, Box<Op>),
    Div(Box<Op>, Box<Op>),
    // eventually add to grammar
    // Min(Box<Op>, Box<Op>),
    // Max(Box<Op>, Box<Op>),
    Exp(Box<Op>, Box<Op>),
    Variable(String),
    Constant(f32),
}

// annotation used in forwards backwards path
#[derive(Debug)]
pub struct ValGrad {
    val: f32,
    grad: f32,
}

// calculate output val and gradient with respect to a given input
fn eval_grad(exp: &Box<Op>, binding: &HashMap<&str, f32>, respect: &str) -> ValGrad {
    return match exp.as_ref() {
        Op::Add(l, r) => {
            let lvg = eval_grad(l, &binding, respect);
            let rvg = eval_grad(r, &binding, respect);
            ValGrad {
                val: lvg.val + rvg.val,
                grad: lvg.grad + rvg.grad,
            }
        }
        Op::Sub(l, r) => {
            let lvg = eval_grad(l, &binding, respect);
            let rvg = eval_grad(r, &binding, respect);
            ValGrad {
                val: lvg.val - rvg.val,
                grad: lvg.grad - rvg.grad,
            }
        }
        Op::Mul(l, r) => {
            let lvg = eval_grad(l, &binding, respect);
            let rvg = eval_grad(r, &binding, respect);
            ValGrad {
                val: lvg.val * rvg.val,
                grad: lvg.grad * rvg.val + rvg.grad * lvg.val,
            }
        }
        Op::Div(l, r) => {
            let lvg = eval_grad(l, &binding, respect);
            let rvg = eval_grad(r, &binding, respect);
            ValGrad {
                val: lvg.val + rvg.val,
                grad: ((lvg.grad * rvg.val) - (rvg.grad * lvg.val)) / (lvg.val * lvg.val),
            }
        }
        Op::Variable(s) => ValGrad {
            val: *binding.get(s.as_str()).unwrap(),
            grad: if s == respect { 1f32 } else { 0f32 },
        },
        Op::Constant(v) => ValGrad {
            val: *v,
            grad: 0f32,
        },
        // exp is messy with ln, cutting for now
        _ => unimplemented!(),
    };
}

fn eval(exp: &Box<Op>, binding: &HashMap<&str, f32>) -> f32 {
    return match exp.as_ref() {
        Op::Add(l, r) => eval(l, &binding) + eval(r, &binding),
        Op::Sub(l, r) => eval(l, &binding) - eval(r, &binding),
        Op::Mul(l, r) => eval(l, &binding) * eval(r, &binding),
        Op::Div(l, r) => eval(l, &binding) / eval(r, &binding),
        Op::Exp(l, r) => eval(l, &binding).powf(eval(r, &binding)),
        Op::Variable(s) => *binding.get(s.as_str()).unwrap(),
        Op::Constant(v) => *v,
    };
}

// simple recursive descent without error checking
// should eventually add parens and function style ops(extension of name)
// parsing does not bind properly
// also needs to ignore spaces
fn parse_exp(s: &String, start: usize) -> (Box<Op>, usize) {
    let mut end = start;

    let mut c = s.as_bytes()[end];
    let mut acc = String::from("");
    while c.is_ascii_alphabetic() {
        acc += (c as char).to_string().as_str();
        end += 1;
        if end >= s.len() {
            break;
        }
        c = s.as_bytes()[end];
    }

    let terminal: Box<Op>;

    // first value must be constant or variable
    if acc.len() != 0 {
        terminal = Box::new(Op::Variable(acc));
    } else {
        while c.is_ascii_digit() {
            acc += (c as char).to_string().as_str();
            end += 1;
            if end >= s.len() {
                break;
            }
            c = s.as_bytes()[end];
        }
        let val: f32 = acc.parse().unwrap();
        terminal = Box::new(Op::Constant(val));
    }

    // base case
    if end >= s.len() {
        return (terminal, end);
    }

    // general case
    let op_c = s.as_bytes()[end];
    let (right, right_end) = parse_exp(&s, end + 1);

    // parse op and construct
    let op = match op_c as char {
        '+' => Op::Add(terminal, right),
        '-' => Op::Sub(terminal, right),
        '*' => Op::Mul(terminal, right),
        '/' => Op::Div(terminal, right),
        '^' => Op::Exp(terminal, right), // this is not actually what exp should be, that's a unary
        // op
        _ => panic!("not supported op"),
    };

    return (Box::new(op), right_end);
}

fn main() {
    let (simple, _) = parse_exp(&String::from("3555*long+2^x"), 0);
    let val_simple = eval(&simple, &HashMap::from([("x", 3f32), ("long", 2f32)]));
    assert!(val_simple == 35550f32);

    let (f, _) = parse_exp(&String::from("x+y*y"), 0);
    let bound = HashMap::from([("x", 2f32), ("y", 6f32)]);
    let dfdx = eval_grad(&f, &bound, "x");
    let dfdy = eval_grad(&f, &bound, "y");

    assert!(dfdx.val == 38f32);
    assert!(dfdx.grad == 1f32);

    assert!(dfdy.val == 38f32);
    assert!(dfdy.grad == 12f32);
}
