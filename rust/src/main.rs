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

// simple recursive descent on expression
// should add parens to force binding
fn parse_exp(s: &String, start: usize) -> (Box<Op>, usize) {
    println!("Calling parse on {}:{}", s, start);
    // no error handling currently
    let mut end = start;

    // base case, variable or constant
    let mut c = s.as_bytes()[end];
    let mut acc = String::from("");
    while c.is_ascii_alphabetic() && end < s.len() {
        acc += (c as char).to_string().as_str();
        end += 1;
        c = s.as_bytes()[end];
    }

    if acc.len() != 0 {
        return (Box::new(Op::Variable(acc)), end);
    } else {
        while c.is_ascii_digit() && end < s.len() {
            acc += (c as char).to_string().as_str();
            end += 1;
            c = s.as_bytes()[end];
        }
        if acc.len() != 0 {
            let val: f32 = acc.parse().unwrap();
            return (Box::new(Op::Constant(val)), end);
        }
    }

    // general case
    let (left, left_end) = parse_exp(&s, start);
    let op_c = s.as_bytes()[left_end];
    let (right, right_end) = parse_exp(&s, left_end + 1);

    // parse op and construct
    let op = match op_c as char {
        '+' => Op::Add(left, right),
        '-' => Op::Sub(left, right),
        '*' => Op::Mul(left, right),
        '/' => Op::Div(left, right),
        '^' => Op::Exp(left, right),
        _ => panic!("not supported op"),
    };

    return (Box::new(op), right_end);
}

fn main() {
    let (ast, _) = parse_exp(&String::from("1+2+3+4"), 0);
    println!("ast: {:?}", ast);
    println!("Hello, world!");
}
