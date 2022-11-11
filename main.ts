// Copyright (c) HashiCorp, Inc
// SPDX-License-Identifier: MPL-2.0
import { Construct } from "constructs";
import { App, TerraformStack } from "cdktf";
import { AwsProvider } from "@cdktf/provider-aws/lib/provider";
import { EcsCluster } from "@cdktf/provider-aws/lib/ecs-cluster";

const region = "us-west-1";
const profile = "forecapp";
const name = "forecapp-api";
const tags = {
  project: "forecapp-api",
};

class MyStack extends TerraformStack {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    new AwsProvider(this, "aws", { region, profile });

    new EcsCluster(this, `ecs-${name}`, {
      name,
      tags,
    });
  }
}

const app = new App();
new MyStack(app, "forecapp-api");
app.synth();
